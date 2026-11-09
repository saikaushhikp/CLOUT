"""P4: observation-consistent world sampler, history replay, batched simulator and rollout planner."""
import copy
import dataclasses
import random
from decimal import Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import Allocation
from dumbocracy.electorate import generate_electorate
from dumbocracy.game import Allocate, GameEngine, observation_for
from dumbocracy.game.actions import validate
from dumbocracy.game.ledger import Ledger
from dumbocracy.sectors import archetype_means
from dumbocracy.simulate import simulate_world
from dumbocracy.strategies.planner import candidate_allocations, default_allocation, plan_round
from dumbocracy.strategies.worlds import (
    WorldBatch, observed_allocations, public_sizes, replay_history, sample_worlds, scaled_sizes,
)

N = 2000
B = (0.3, 0.2, 0.2, 0.15, 0.15)


@pytest.fixture(scope="module")
def config():
    return GameConfig(seed=21, rounds=4, device="cpu", electorate=ElectorateConfig(num_voters=N))


@pytest.fixture(scope="module")
def electorate(config):
    return generate_electorate(config)


@pytest.fixture(scope="module")
def survey_state(config, electorate):
    return GameEngine.new_game(config, electorate=electorate, game_id="w")


def _ok(state, action):
    step = GameEngine.apply(state, "p0", action)
    assert step.accepted, (step.error_code, step.message)
    return step.state


PLAYED = [Allocation((3, 3, 2, 2), 5, (True, False, False, False)), Allocation((0, 10, 0, 0), 0, (False, True, True, False))]


@pytest.fixture(scope="module")
def round3_state(survey_state):
    state = _ok(survey_state, {"type": "commit", "manifesto": list(B)})
    for a in PLAYED:
        state = _ok(state, {"type": "allocate", "workers_by_region": list(a.workers_by_region),
                            "extra_workers": a.extra_workers, "media_regions": list(a.media_regions)})
    return state


ASSIGN = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2], [2, 0, 5, 1]])


def test_worlds_respect_public_sizes_and_assignment(survey_state, config):
    obs = observation_for(survey_state, "p0")
    worlds = sample_worlds(obs, ASSIGN, config, seed=3)
    means = archetype_means(dtype=torch.float32)
    assert len(worlds) == 3
    for w, sigma in zip(worlds, ASSIGN):
        assert w.sizes.tolist() == public_sizes(obs)
        assert w.archetype_of_region.tolist() == sigma.tolist()
        for r in range(4):
            region_mean = w.preferences[w.region_of == r].mean(0)
            assert float((region_mean - means[sigma[r]]).abs().sum()) < 0.03
        assert sorted(w.personality_of_region.tolist()) == [0, 1, 2, 3]
        # the census-matched graph reproduces the public edge counts (sampling noise only)
        counts = w.census()
        for got, want in zip(counts["regions"], obs["regions"]):
            assert abs(got["intra_edges"] - want["intra_edges"]) < 0.05 * want["intra_edges"]


def test_worlds_rescaled_keep_proportions_and_floor(survey_state, config):
    obs = observation_for(survey_state, "p0")
    sizes = public_sizes(obs)
    worlds = sample_worlds(obs, ASSIGN[:1], config, num_voters=1000, seed=3)
    small = worlds[0].sizes.tolist()
    assert sum(small) == 1000 and min(small) >= 150
    for a, b in zip(small, sizes):
        assert abs(a / 1000 - b / N) < 0.002
    total_edges = worlds[0].graph.num_edges
    assert abs(total_edges - survey_state.electorate.graph.num_edges / 2) < 0.06 * survey_state.electorate.graph.num_edges / 2
    for raw, n in (([5000, 1500, 1500, 2000], 1000), ([1500, 1500, 1500, 5500], 999)):
        out = scaled_sizes(raw, n)
        assert sum(out) == n and min(out) >= 150


def test_worlds_deterministic_per_seed_and_prior_graph(survey_state, config):
    obs = observation_for(survey_state, "p0")
    a = sample_worlds(obs, ASSIGN[:2], config, num_voters=800, seed=9)
    b = sample_worlds(obs, ASSIGN[:2], config, num_voters=800, seed=9)
    c = sample_worlds(obs, ASSIGN[:2], config, num_voters=800, seed=10)
    assert all(torch.equal(x.preferences, y.preferences) for x, y in zip(a, b))
    assert not torch.equal(a[0].preferences, c[0].preferences)
    prior = sample_worlds(obs, ASSIGN[:1], config, num_voters=800, seed=9, graph="prior")
    assert prior[0].graph.num_edges > 0


def test_replay_advances_rounds_and_matches_history(round3_state, config):
    obs = observation_for(round3_state, "p0")
    assert obs["round"] == 3
    assert observed_allocations(obs) == PLAYED
    spent = sum(Decimal(a["cost"]) for a in obs["allocations"])
    assert spent == config.budget - Decimal(obs["treasury"])
    world = sample_worlds(obs, ASSIGN[:1], config, num_voters=1000, seed=1)[0]
    v2 = replay_history(world, obs, config, seed=4)
    assert not torch.allclose(v2, world.preferences)
    # same as stepping the two observed allocations by hand
    batch = WorldBatch([world], config)
    one = copy.deepcopy(obs)
    one["allocations"] = obs["allocations"][:1]
    v1 = batch.replay(one, seed=4)
    assert not torch.allclose(v1, v2)
    assert torch.allclose(replay_history(world, obs, config, seed=4), v2)
    # pulled toward the manifesto: similarity to b increases relative to v(0)
    b = torch.tensor(B)
    assert float((v2 - b).abs().sum(1).mean()) < float((world.preferences - b).abs().sum(1).mean())
    # before commit, replay is the identity
    pre = dict(obs, manifesto=None, allocations=[])
    assert torch.equal(replay_history(world, pre, config), world.preferences)


def test_world_batch_matches_simulate_world(config, electorate):
    """The batched campaign/dynamics reproduce simulate_world's expected share (statistically)."""
    manifesto = torch.tensor(B)
    plan = Allocation((5, 5, 5, 5), 10, (True, False, False, False))
    policy = lambda ctx: plan  # noqa: E731
    reference = [simulate_world(electorate, dataclasses.replace(config, seed=s), manifesto, policy).share_trajectory[-1]
                 for s in range(4)]
    batch = WorldBatch([electorate], config)
    v = batch.preferences.unsqueeze(0).expand(4, -1, -1).contiguous()
    airings = torch.zeros((4, 4), dtype=torch.int64)
    for t in range(config.rounds):
        v = batch.step(v, [plan] * 4, airings, manifesto, torch.Generator().manual_seed(100 + t))
        airings = airings + torch.tensor([[1, 0, 0, 0]] * 4)
    ours = float(batch.shares(v, manifesto).mean())
    assert abs(ours - sum(reference) / len(reference)) < 0.01


def _random_round_observation(base_obs, rng, config):
    obs = copy.deepcopy(base_obs)
    obs["round"] = rng.randint(1, config.rounds)
    obs["treasury"] = f"{Decimal(rng.randint(0, 50000)) / 100:.2f}"
    if rng.random() < 0.2:
        obs["treasury"] = f"{Decimal(rng.randint(0, 3000)) / 100:.2f}"
    obs["media_airings"] = {r: rng.randint(0, 4) for r in ("N", "S", "E", "W")}
    hired = rng.choice([0, 0, rng.randint(1, 60)])   # R16: permanent hires so far
    free = config.workers.free_workers
    obs["workers"] = {"free": free, "hired": hired, "available": free + hired}
    return obs


def test_candidates_valid_and_affordable_property(round3_state, config):
    """R16: every candidate is valid (≤ free + earlier hires + new hires) and affordable, accepted by the engine."""
    base = observation_for(round3_state, "p0")
    assert base["workers"] == {"free": 10, "hired": 5, "available": 15}   # PLAYED hired 5 in round 1
    rng = random.Random(0)
    player = round3_state.player("p0")
    for _ in range(300):
        obs = _random_round_observation(base, rng, config)
        hired = obs["workers"]["hired"]
        cands = candidate_allocations(obs, config)
        assert 1 <= len(cands) <= 20
        assert len(set(cands)) == len(cands)
        state = round3_state.with_player(dataclasses.replace(player, ledger=Ledger(Decimal(obs["treasury"])),
                                                             hired_workers=hired))
        for a in cands:
            a.validate(config.workers.free_workers, hired)
            # candidates put every available worker to work (free + earlier hires + new hires)
            assert sum(a.workers_by_region) == config.workers.free_workers + hired + a.extra_workers
            action = Allocate(tuple(a.workers_by_region), a.extra_workers, tuple(a.media_regions))
            assert validate(state, state.player("p0"), action) is None
        # the default continuation is affordable and valid too
        d = default_allocation(public_sizes(obs), Decimal(obs["treasury"]), 2, [0, 0, 0, 0], config, hired)
        assert validate(state, state.player("p0"), Allocate(d.workers_by_region, d.extra_workers, d.media_regions)) is None
        # ... and without counting the earlier hires it would never over-assign either
        if hired:
            too_many = Allocate((config.workers.free_workers + hired + 1, 0, 0, 0), 0, (False,) * 4)
            assert validate(state, state.player("p0"), too_many).code == "INVALID_ALLOCATION"


def test_candidates_diverse_with_budget(round3_state, config):
    obs = observation_for(round3_state, "p0")
    cands = candidate_allocations(obs, config)
    assert len(cands) >= 10
    assert any(any(a.media_regions) for a in cands) and any(a.extra_workers > 0 for a in cands)
    assert any(a.extra_workers == 0 and not any(a.media_regions) for a in cands)
    broke = dict(obs, treasury="0.00")
    assert all(a.extra_workers == 0 and not any(a.media_regions) for a in candidate_allocations(broke, config))


def test_plan_round_deterministic_and_returns_candidate(round3_state, config):
    obs = observation_for(round3_state, "p0")
    worlds = sample_worlds(obs, ASSIGN, config, num_voters=1000, seed=5)
    a1, d1 = plan_round(obs, worlds, obs["manifesto"], config, seed=7)
    a2, d2 = plan_round(obs, worlds, obs["manifesto"], config, seed=7)
    assert a1 == a2 and d1["scores"] == d2["scores"]
    assert a1 in d1["candidates"]
    base = candidate_allocations(obs, config)
    assert d1["candidates"][:len(base)] == base  # + value-guided candidates from the same worlds
    treasury = Decimal(obs["treasury"])
    for a in d1["candidates"]:
        a.validate(config.workers.free_workers, obs["workers"]["hired"])
        assert a.extra_workers * config.workers.extra_worker_cost <= treasury
    _, d4 = plan_round(obs, worlds, obs["manifesto"], config, seed=7, value_guided=False)
    assert d4["candidates"] == base
    assert d1["num_worlds"] == 3 and len(d1["scores"]) == d1["num_candidates"]
    assert d1["best_score"] == max(d1["scores"])
    a3, d3 = plan_round(obs, worlds, obs["manifesto"], config, seed=7, rollout="none")
    assert a3 in d3["candidates"]
