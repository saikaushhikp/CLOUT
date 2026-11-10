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
from dumbocracy.strategies.planner import allocation_cost, candidate_allocations, default_allocation, plan_round
from dumbocracy.strategies.worlds import (
    BatchWorkers, WorldBatch, observed_allocations, public_sizes, replay_history, sample_worlds, scaled_sizes,
    team_by_region,
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


PLAYED = [Allocation((1, 1, 0, 0), (True, False, False, False)), Allocation((0, 1, 0, 0), (False, True, True, False))]


@pytest.fixture(scope="module")
def round3_state(survey_state):
    state = _ok(survey_state, {"type": "commit", "manifesto": list(B)})
    for a in PLAYED:
        state = _ok(state, {"type": "allocate", "hires_by_region": list(a.hires_by_region),
                            "media_regions": list(a.media_regions)})
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
    # the replayed worlds carry the workers: 3 bought in total (scaled to the 1000-voter world: ~1.5)
    v_w, workers = WorldBatch([world], config).replay_state(obs, seed=4)
    assert torch.equal(v_w, v2) and workers.plans == 1
    assert 0 <= int(workers.pinned.sum()) <= 3
    assert torch.equal(v_w[workers.pinned[0]], b.to(v_w.dtype).expand(int(workers.pinned.sum()), -1))
    assert team_by_region(obs) == [1, 2, 0, 0]
    # before commit, replay is the identity
    pre = dict(obs, manifesto=None, allocations=[])
    assert torch.equal(replay_history(world, pre, config), world.preferences)


def test_world_batch_matches_simulate_world(config, electorate):
    """The batched P8 campaign/dynamics reproduce simulate_world's expected share and cascade (statistically)."""
    manifesto = torch.tensor(B)
    plans = [Allocation((2, 2, 1, 1), (True, False, False, False))] + [Allocation((0, 0, 0, 0), (True, False, False, False))] * 3
    policy = lambda ctx: plans[ctx.round]  # noqa: E731
    runs = [simulate_world(electorate, dataclasses.replace(config, seed=s), manifesto, policy) for s in range(6)]
    reference = sum(r.share_trajectory[-1] for r in runs) / len(runs)
    ref_pinned = sum(int(r.workers.pinned.sum()) for r in runs) / len(runs)
    batch = WorldBatch([electorate], config)
    C = 6
    v = batch.preferences.unsqueeze(0).expand(C, -1, -1).contiguous()
    airings = torch.zeros((C, 4), dtype=torch.int64)
    workers = None
    for t in range(config.rounds):
        v, workers = batch.step(v, [plans[t]] * C, airings, manifesto, torch.Generator().manual_seed(100 + t), workers)
        airings = airings + torch.tensor([[1, 0, 0, 0]] * C)
    ours = float(batch.shares(v, manifesto).mean())
    assert abs(ours - reference) < 0.02, (ours, reference)
    assert ref_pinned == 6 and bool((workers.pinned.sum(1) == 6).all())
    # pinned workers sit exactly at b, zealots never move
    for c in range(C):
        assert torch.equal(v[c][workers.pinned[c]], manifesto.expand(6, -1))
    z = batch.zealot
    assert torch.equal(v[:, z], batch.preferences[z].expand(C, -1, -1))


def test_world_batch_picks_follow_r22(config, electorate):
    batch = WorldBatch([electorate, electorate], config)
    plans = [Allocation((3, 0, 1, 0)), Allocation((0, 2, 0, 2))]
    v = batch.preferences.unsqueeze(0).expand(2, -1, -1).contiguous()
    _, workers = batch.step(v, plans, torch.zeros((2, 4), dtype=torch.int64), torch.tensor(B), torch.Generator().manual_seed(1))
    region = batch.local_region
    world = torch.arange(2 * batch.n) // batch.n
    for c, a in enumerate(plans):
        for w in range(2):
            for r in range(4):
                assert int((workers.pinned[c] & (region == r) & (world == w)).sum()) == a.hires_by_region[r]
        assert bool((batch.graph.degree[workers.pinned[c]] >= 3).all()) and not bool(batch.zealot[workers.pinned[c]].any())
    # rescaled worlds scale the hires (N/n = 2): 4 hires -> 2 per world on average
    small = sample_worlds(observation_for(GameEngine.new_game(config, electorate=electorate), "p0"), ASSIGN, config,
                          num_voters=1000, seed=3)
    sb = WorldBatch(small, config)
    assert sb.hire_scale == pytest.approx(0.5)
    counts = []
    for s in range(20):
        _, w = sb.step(sb.preferences.unsqueeze(0), [Allocation((3, 0, 0, 0))], torch.zeros((1, 4), dtype=torch.int64),
                       torch.tensor(B), torch.Generator().manual_seed(s))
        counts.append(int(w.pinned.sum()))
    assert all(3 <= x <= 6 for x in counts) and abs(sum(counts) / 20 - 4.5) < 1.0


def _random_round_observation(base_obs, rng, config):
    obs = copy.deepcopy(base_obs)
    obs["round"] = rng.randint(1, config.rounds)
    obs["treasury"] = f"{Decimal(rng.randint(0, 50000)) / 100:.2f}"
    if rng.random() < 0.2:
        obs["treasury"] = f"{Decimal(rng.randint(0, 3000)) / 100:.2f}"
    obs["media_airings"] = {r: rng.randint(0, 4) for r in ("N", "S", "E", "W")}
    return obs


def test_candidates_valid_and_affordable_property(round3_state, config):
    """Every candidate is valid and affordable and accepted by the engine (P8 hires + media)."""
    base = observation_for(round3_state, "p0")
    assert base["workers"] == {"by_region": {"N": 1, "S": 2, "E": 0, "W": 0}, "total": 3}
    rng = random.Random(0)
    player = round3_state.player("p0")
    for _ in range(300):
        obs = _random_round_observation(base, rng, config)
        cands = candidate_allocations(obs, config)
        assert 1 <= len(cands) <= 20
        assert len(set(cands)) == len(cands)
        state = round3_state.with_player(dataclasses.replace(player, ledger=Ledger(Decimal(obs["treasury"]))))
        for a in cands:
            a.validate()
            assert allocation_cost(a, public_sizes(obs), config) <= Decimal(obs["treasury"])
            action = Allocate(tuple(a.hires_by_region), tuple(a.media_regions))
            assert validate(state, state.player("p0"), action) is None
        # the default continuation is affordable and valid too
        d = default_allocation(public_sizes(obs), Decimal(obs["treasury"]), 2, [0, 0, 0, 0], config)
        assert validate(state, state.player("p0"), Allocate(d.hires_by_region, d.media_regions)) is None


def test_candidates_diverse_with_budget(round3_state, config):
    obs = observation_for(round3_state, "p0")
    cands = candidate_allocations(obs, config)
    assert len(cands) >= 10
    assert any(any(a.media_regions) for a in cands) and any(a.total_hires > 0 for a in cands)
    assert any(a.total_hires == 0 and not any(a.media_regions) for a in cands)
    assert any(a.total_hires == 0 and any(a.media_regions) for a in cands)            # media-only
    assert any(sum(1 for h in a.hires_by_region if h) == 1 for a in cands)              # focus on one region
    broke = dict(obs, treasury="0.00")
    assert candidate_allocations(broke, config) == [Allocation()]


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
        a.validate()
        assert allocation_cost(a, public_sizes(obs), config) <= treasury
    _, d4 = plan_round(obs, worlds, obs["manifesto"], config, seed=7, value_guided=False)
    assert d4["candidates"] == base
    assert d1["num_worlds"] == 3 and len(d1["scores"]) == d1["num_candidates"]
    assert d1["best_score"] == max(d1["scores"])
    a3, d3 = plan_round(obs, worlds, obs["manifesto"], config, seed=7, rollout="none")
    assert a3 in d3["candidates"]


def test_plan_round_is_fast_at_10k():
    """Rollout planner decision at N = 10k (players use 8 worlds x 1600 voters) stays under 1 s."""
    import time
    from dumbocracy.strategies.players import BayesPlanner
    cfg = GameConfig(seed=5, rounds=5, device="cpu")
    state = GameEngine.new_game(cfg)
    state = _ok(state, {"type": "commit", "manifesto": list(B)})
    state = _ok(state, {"type": "allocate", "hires_by_region": [2, 0, 1, 0], "media_regions": [True, False, False, False]})
    obs = observation_for(state, "p0")
    player = BayesPlanner(seed=1, round_query_gain=None)
    player.allocation(obs)          # warm-up
    start = time.perf_counter()
    allocation, diag = player.allocation(obs)
    assert time.perf_counter() - start < 1.0, diag["seconds"]
    allocation.validate()


@pytest.mark.parametrize("fee", ["1.00", "50.00", "137.50", "600.00"])
def test_candidates_and_default_affordable_at_any_fee(round3_state, config, fee):
    """Strategies must never exceed the treasury whatever the worker fee is."""
    cfg = dataclasses.replace(config, workers=dataclasses.replace(config.workers, worker_cost=Decimal(fee)))
    base = observation_for(round3_state, "p0")
    rng = random.Random(1)
    for _ in range(100):
        obs = _random_round_observation(base, rng, cfg)
        treasury = Decimal(obs["treasury"])
        cands = candidate_allocations(obs, cfg)
        assert all(allocation_cost(a, public_sizes(obs), cfg) <= treasury for a in cands)
        if treasury >= Decimal(fee) + 200:   # enough for a worker and an airing: the set stays diverse
            assert any(a.total_hires > 0 for a in cands) and any(any(a.media_regions) for a in cands)
            assert len(cands) >= 5
        for left in (1, 2, 4):
            d = default_allocation(public_sizes(obs), treasury, left, [0, 0, 0, 0], cfg)
            assert allocation_cost(d, public_sizes(obs), cfg) <= treasury


@pytest.mark.parametrize("fee", ["50.00", "137.50"])
def test_scripted_players_never_overspend(fee):
    from dumbocracy.strategies.players import RandomPlayer, Top3Player, play_game
    cfg = GameConfig(seed=31, rounds=3, device="cpu", electorate=ElectorateConfig(num_voters=1000),
                     workers=dataclasses.replace(GameConfig().workers, worker_cost=Decimal(fee)))
    for player in (Top3Player(config=cfg), RandomPlayer(seed=2)):
        final, log = play_game(GameEngine.new_game(cfg, game_id="fee"), player)   # strict: a rejection raises
        assert final.phase == "DONE" and all(ok for *_, ok, _ in log)
