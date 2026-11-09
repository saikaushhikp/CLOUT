"""Bayesian belief over σ (strategies/bayes.py; spec §2.3, §6.2–§6.3, §10.1)."""
import json
import math
import time

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.electorate import generate_electorate
from dumbocracy.information.survey import global_top3_pct, query_moe, sample_query
from dumbocracy.rng import dirichlet
from dumbocracy.sectors import ARCHETYPES, REGIONS, SECTORS, archetype_means
from dumbocracy.strategies.bayes import (
    ASSIGNMENTS, AnswerModel, Posterior, best_query, build_answer_model, expected_information_gain,
    posterior_from_observation, posterior_region_mean,
)

SEEDS = range(20)
KS = (1, 2, 3, 6, 10)


@pytest.fixture(scope="module")
def model() -> AnswerModel:
    return build_answer_model()


@pytest.fixture(scope="module")
def worlds():
    out = []
    for seed in SEEDS:
        e = generate_electorate(GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=10000)))
        out.append((e.preferences, e.region_of, [int(x) for x in e.sizes.tolist()], tuple(e.archetype_of_region.tolist())))
    return out


def base_observation(prefs, sizes) -> dict:
    return {"phase": "SURVEY", "round": 0,
            "regions": [{"id": r, "voters": n} for r, n in zip(REGIONS, sizes)],
            "free_stats": [{"round": 0, "top3_pct": global_top3_pct(prefs)}], "queries": []}


def answer(prefs, region_of, triple, generator) -> dict:
    """A sampled (R12) answer exactly as the engine builds it: fresh 10% sample of the region."""
    a, b, r = triple
    yes, m, n = sample_query(prefs, region_of, SECTORS.index(a), SECTORS.index(b), REGIONS.index(r), 0.10, generator)
    return {"round": 0, "phase": "SURVEY", "sector_a": a, "sector_b": b, "region": r, "pct_b_over_a": 100.0 * yes / m,
            "sample_size": m, "moe": query_moe(yes, m, n), "cost": "5.00"}


def true_index(sigma) -> int:
    return int((ASSIGNMENTS == torch.tensor(sigma)).all(1).nonzero().item())


# ------------------------------------------------------------------------------------------ basics
def test_assignments_are_all_injective_maps():
    assert ASSIGNMENTS.shape == (360, 4) and ASSIGNMENTS.dtype == torch.int64
    assert len({tuple(row) for row in ASSIGNMENTS.tolist()}) == 360
    assert all(len(set(row)) == 4 for row in ASSIGNMENTS.tolist())
    assert ASSIGNMENTS.tolist() == sorted(ASSIGNMENTS.tolist())


def test_empty_observation_gives_uniform_prior():
    post = posterior_from_observation({})
    assert torch.allclose(post.probs, torch.full((360,), 1 / 360, dtype=torch.float64))
    assert math.isclose(post.entropy(), math.log(360))
    assert torch.allclose(post.region_marginals(), torch.full((4, 6), 1 / 6, dtype=torch.float64))
    assert not post.inconsistent
    json.dumps(post.to_public_dict())


def test_answer_model_consistency(model):
    qp = model.query_p
    assert qp.shape == (6, 5, 5) and model.top3.shape == (6, 5)
    assert torch.all(torch.diagonal(qp, dim1=1, dim2=2) == 0)
    off = ~torch.eye(5, dtype=torch.bool)
    assert torch.allclose((qp + qp.transpose(1, 2))[:, off], torch.ones(6, 20, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(model.top3.sum(1), torch.full((6,), 3.0, dtype=torch.float64))
    balanced = qp[5][off]  # A5 "Balanced middle": symmetric answers
    assert torch.all((balanced - 0.5).abs() < 0.01)
    assert torch.all((model.top3[5] - 0.6).abs() < 0.01)
    assert build_answer_model() is model  # cached


def test_answer_model_matches_direct_simulation(model):
    g = torch.Generator().manual_seed(123)
    n = 100_000
    v = dirichlet(60.0 * archetype_means(dtype=torch.float64)[0].expand(n, 5).contiguous(), g)
    se = 4 * math.sqrt(0.25 / n) + 4 * math.sqrt(0.25 / 200_000)
    for a in range(5):
        for b in range(5):
            if a != b:
                assert abs(float((v[:, b] > v[:, a]).double().mean()) - float(model.query_p[0, a, b])) < se
    top = global_top3_pct(v)
    for s, name in enumerate(SECTORS):
        assert abs(top[name] / 100 - float(model.top3[0, s])) < se


def test_posterior_methods(model, worlds):
    prefs, region_of, sizes, sigma = worlds[0]
    post = posterior_from_observation(base_observation(prefs, sizes), model)
    assert math.isclose(float(post.probs.sum()), 1.0, abs_tol=1e-12)
    marg = post.region_marginals()
    assert marg.shape == (4, 6) and torch.allclose(marg.sum(1), torch.ones(4, dtype=torch.float64))
    top = post.top(5)
    assert len(top) == 5 and top[0][0] >= top[-1][0] and tuple(post.map_assignment().tolist()) == top[0][1]
    draws = post.sample(64, torch.Generator().manual_seed(0))
    assert draws.shape == (64, 4)
    assert all(post.probs[true_index(tuple(row))] > 0 for row in draws.tolist())
    mean = posterior_region_mean(post, model)
    assert mean.shape == (4, 5) and torch.allclose(mean.sum(1), torch.ones(4, dtype=torch.float64))
    public = post.to_public_dict()
    json.dumps(public)
    assert set(public["marginals"]) == set(REGIONS) and len(public["top"]) == 5
    assert public["archetypes"][0]["name"] == ARCHETYPES[0].name


def test_inconsistent_observation_falls_back_to_prior(model):
    obs = {"regions": [{"id": r, "voters": 2500} for r in REGIONS],
           "queries": [{"round": 0, "sector_a": "Agri", "sector_b": "Industry", "region": "N", "pct_b_over_a": 0.0},
                       {"round": 0, "sector_a": "Industry", "sector_b": "Agri", "region": "N", "pct_b_over_a": 0.0}]}
    post = posterior_from_observation(obs, model)
    assert post.inconsistent
    assert torch.allclose(post.probs, torch.full((360,), 1 / 360, dtype=torch.float64))


def test_free_stats_option(model, worlds):
    prefs, _, sizes, _ = worlds[0]
    obs = base_observation(prefs, sizes)
    assert math.isclose(posterior_from_observation(obs, model, use_free_stats="none").entropy(), math.log(360))
    assert posterior_from_observation(obs, model).entropy() < math.log(360) - 0.5
    with pytest.raises(ValueError):
        posterior_from_observation(obs, model, use_free_stats="bogus")


# ------------------------------------------------------------------------------- information gain
def test_information_gain_properties(model):
    sizes = [2500, 2500, 2500, 2500]
    start = time.perf_counter()
    gains = expected_information_gain(Posterior.uniform(), model, sizes)
    assert time.perf_counter() - start < 1.0
    assert len(gains) == 40 and all(g >= 0 for g, _ in gains)
    assert all(gains[i][0] >= gains[i + 1][0] for i in range(39))
    assert all(SECTORS.index(a) < SECTORS.index(b) for _, (a, b, _) in gains)
    assert gains[0][0] <= math.log(6) + 1e-6
    point = torch.zeros(360, dtype=torch.float64)
    point[17] = 1.0
    assert all(g == 0.0 for g, _ in expected_information_gain(Posterior(point), model, sizes))
    # later rounds are noisier, hence less informative
    later = dict((t, g) for g, t in expected_information_gain(Posterior.uniform(), model, sizes, round_index=4))
    assert all(later[t] < g for g, t in gains)
    # R12: repeats allowed — by default nothing is excluded; the optional ``asked`` excludes both orientations
    first = best_query(Posterior.uniform(), model, sizes)
    assert first == gains[0][1]
    second = best_query(Posterior.uniform(), model, sizes, asked={(first[1], first[0], first[2])})
    assert second != first
    # a smaller sample is less informative
    small = dict((t, g) for g, t in expected_information_gain(Posterior.uniform(), model, sizes, sample_fraction=0.02))
    census = dict((t, g) for g, t in expected_information_gain(Posterior.uniform(), model, sizes, sample_fraction=1.0))
    assert all(small[t] < g <= census[t] + 1e-9 for g, t in gains if g > 1e-3)


def test_likelihood_uses_each_answers_sample_size(model):
    sizes = [2500, 2500, 2500, 2500]
    obs = {"regions": [{"id": r, "voters": n} for r, n in zip(REGIONS, sizes)], "queries": []}
    q = {"round": 0, "sector_a": "Agri", "sector_b": "Industry", "region": "N", "pct_b_over_a": 40.0}
    small = posterior_from_observation({**obs, "queries": [{**q, "sample_size": 10}]}, model)
    large = posterior_from_observation({**obs, "queries": [{**q, "sample_size": 1000}]}, model)
    default = posterior_from_observation({**obs, "queries": [q]}, model)  # falls back to m = 250
    explicit = posterior_from_observation({**obs, "queries": [{**q, "sample_size": 250}]}, model)
    assert large.entropy() < default.entropy() < small.entropy()
    assert torch.allclose(default.probs, explicit.probs)
    # each repeat is an independent observation; pooled_* fields are ignored (no double counting)
    pooled = {**q, "sample_size": 250, "pooled_pct": 40.0, "pooled_sample_size": 500, "repeat_index": 2}
    assert torch.allclose(posterior_from_observation({**obs, "queries": [pooled]}, model).probs, explicit.probs)
    twice = posterior_from_observation({**obs, "queries": [{**q, "sample_size": 250}] * 2}, model)
    assert twice.entropy() < explicit.entropy()


# --------------------------------------------------------------------------------------- recovery
def run_queries(model, prefs, region_of, sizes, policy, seed) -> tuple[Posterior, dict[int, float], dict[int, Posterior]]:
    """Ask max(KS) sampled queries (repeats allowed) by info gain or uniformly at random."""
    sample_gen = torch.Generator().manual_seed(1000 + seed)
    pick_gen = torch.Generator().manual_seed(2000 + seed)
    obs = base_observation(prefs, sizes)
    post = posterior_from_observation(obs, model)
    entropies, posts = {0: post.entropy()}, {}
    for k in range(1, max(KS) + 1):
        if policy == "info":
            triple = best_query(post, model, sizes)
        else:
            options = [t for _, t in expected_information_gain(post, model, sizes)]
            triple = options[int(torch.randint(len(options), (1,), generator=pick_gen))]
        obs["queries"].append(answer(prefs, region_of, triple, sample_gen))
        post = posterior_from_observation(obs, model)
        if k in KS:
            entropies[k], posts[k] = post.entropy(), post
    return post, entropies, posts


def test_true_assignment_keeps_mass_with_free_stat_and_sampled_queries(model, worlds):
    g = torch.Generator().manual_seed(7)
    kept = 0
    for seed, (prefs, region_of, sizes, sigma) in enumerate(worlds):
        obs = base_observation(prefs, sizes)
        sample_gen = torch.Generator().manual_seed(seed)
        for _ in range(6):
            a, b = torch.randperm(5, generator=g)[:2].tolist()
            r = int(torch.randint(4, (1,), generator=g))
            obs["queries"].append(answer(prefs, region_of, (SECTORS[a], SECTORS[b], REGIONS[r]), sample_gen))
        post = posterior_from_observation(obs, model)
        idx = true_index(sigma)
        in_top5 = any(a == sigma for _, a in post.top(5))
        kept += int(in_top5 or float(post.probs[idx]) >= 1e-3)
        assert not post.inconsistent
    print(f"\n[bayes] true σ kept (top-5 or p≥1e-3): {kept}/{len(worlds)}")
    assert kept >= math.ceil(0.95 * len(worlds))


def test_info_gain_queries_recover_sigma_and_beat_random(model, worlds):
    n = len(worlds)
    info = {k: 0.0 for k in (0,) + KS}
    rand = {k: 0.0 for k in (0,) + KS}
    map_hits = {6: 0, 10: 0}
    kept = 0
    for seed, (prefs, region_of, sizes, sigma) in enumerate(worlds):
        _, ent, posts = run_queries(model, prefs, region_of, sizes, "info", seed)
        _, ent_r, _ = run_queries(model, prefs, region_of, sizes, "random", seed)
        for k in info:
            info[k] += ent[k] / n
            rand[k] += ent_r[k] / n
        for k in map_hits:
            map_hits[k] += int(tuple(posts[k].map_assignment().tolist()) == sigma)
        kept += int(float(posts[10].probs[true_index(sigma)]) >= 1e-3)
    print("\n[bayes] mean entropy (nats) after k sampled queries: "
          + ", ".join(f"k={k}: info {info[k]:.4f} / random {rand[k]:.4f}" for k in info)
          + f"; MAP == σ after 6: {map_hits[6]}/{n}, after 10: {map_hits[10]}/{n}")
    assert map_hits[6] >= 0.8 * n and map_hits[10] >= 0.8 * n
    assert kept >= math.ceil(0.95 * n)
    for k in (1, 2, 3):
        assert info[k] < rand[k]
    assert info[2] < 0.5 * info[0] and info[6] < 0.2
