"""P8 campaign core: worker picks (R22), one IC hop per round (R24–R27), pinned workers (R23), media (R28)."""
import dataclasses
import math
from decimal import Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig, MediaConfig, VotingConfig, WorkerConfig
from dumbocracy.dynamics.campaign import (
    Allocation, WorkerState, campaign_round, combine_intensities, conversion_chance, eligible_counts,
    eligible_workers, ic_hop, media_cost, media_exposure, media_intensity, pick_workers, spreader_edge_counts,
    worker_cost,
)
from dumbocracy.dynamics.update import step_preferences
from dumbocracy.electorate import Graph, generate_electorate
from dumbocracy.electorate.attributes import Susceptibilities
from dumbocracy.electorate.electorate import Electorate
from dumbocracy.simulate import resolve_campaign_round, simulate_world


def gen(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def star_graph(leaves):
    return Graph.from_edges(leaves + 1, torch.zeros(leaves, dtype=torch.int64), torch.arange(1, leaves + 1))


REGION_OF = torch.repeat_interleave(torch.arange(4), torch.tensor([100, 200, 300, 400]))
B = torch.tensor([0.3, 0.2, 0.2, 0.15, 0.15], dtype=torch.float64)


def electorate(device="cpu", seed=3, n=5000):
    return generate_electorate(GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=n)), device=device)


def generators(seed, device="cpu"):
    return {name: gen(seed * 10 + i, device) for i, name in enumerate(("workers", "cascade", "media"))}


def toy_electorate(graph, *, gamma=1.0, zealot=()):
    """Single-region electorate on ``graph`` with every voter at B (so P is the same for everyone)."""
    n = graph.num_nodes
    z = torch.zeros(n, dtype=torch.bool)
    z[list(zealot)] = True
    ones = torch.ones(n, dtype=torch.float64)
    sus = Susceptibilities(ones * 0.5, ones * 0.5, ones * gamma, z)
    sizes = torch.tensor([n, 0, 0, 0])
    return Electorate(sizes, torch.zeros(n, dtype=torch.int64), graph, B.repeat(n, 1), sus, torch.arange(4),
                      torch.arange(4), torch.zeros(4), torch.zeros(4, 4), torch.zeros(4, 4), 1.0, False)


SURE = VotingConfig(sensitivity=1000.0, threshold=0.5)    # S = 1 for v = b  ->  P = sigma(500) = 1 exactly
NEVER = VotingConfig(sensitivity=1000.0, threshold=1.0)   # S < 0.96         ->  P = 0 exactly


# ------------------------------------------------------------------------------------- picks (R22)

def test_pick_workers_rules_and_determinism():
    e = electorate(n=2000)
    zealot = e.susceptibility.zealot
    none = torch.zeros(e.num_voters, dtype=torch.bool)
    eligible = eligible_workers(e.graph.degree, zealot, none, 3)
    ids = pick_workers(e.region_of, eligible, [3, 0, 5, 2], gen(1))
    assert [x.numel() for x in ids] == [3, 0, 5, 2]
    flat = torch.cat(ids)
    assert flat.unique().numel() == 10
    for r, x in enumerate(ids):
        assert bool((e.region_of[x] == r).all())
    assert bool((e.graph.degree[flat] >= 3).all()) and not bool(zealot[flat].any())
    again = pick_workers(e.region_of, eligible, [3, 0, 5, 2], gen(1))
    assert all(torch.equal(a, b) for a, b in zip(ids, again))
    other = pick_workers(e.region_of, eligible, [3, 0, 5, 2], gen(2))
    assert not all(torch.equal(a, b) for a, b in zip(ids, other))
    # no repeats across rounds: existing workers are excluded
    taken = none.clone()
    taken[flat] = True
    later = torch.cat(pick_workers(e.region_of, eligible_workers(e.graph.degree, zealot, taken, 3), [3, 3, 3, 3], gen(3)))
    assert not bool(taken[later].any())


def test_pick_workers_rejects_too_few_eligible():
    e = electorate(n=2000)
    eligible = eligible_workers(e.graph.degree, e.susceptibility.zealot, torch.zeros(e.num_voters, dtype=torch.bool), 3)
    counts = eligible_counts(e.region_of, eligible)
    assert all(0 < c <= int(s) for c, s in zip(counts, e.sizes.tolist()))
    pick_workers(e.region_of, eligible, [counts[0], 0, 0, 0], gen(1))       # exactly all is fine
    with pytest.raises(ValueError):
        pick_workers(e.region_of, eligible, [counts[0] + 1, 0, 0, 0], gen(1))
    with pytest.raises(ValueError):
        pick_workers(e.region_of, eligible, [-1, 0, 0, 0], gen(1))
    # degree rule: on a star only the centre has >= 3 friends
    star = star_graph(5)
    ok = eligible_workers(star.degree, torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool), 3)
    assert ok.tolist() == [True] + [False] * 5


# ------------------------------------------------------------------------- one hop per round (R24)

def lollipop():
    """0 is the only voter with >= 3 friends: 0-{1,2,3}, then the path 3-4-5-6-7-8-9."""
    src = torch.tensor([0, 0, 0, 3, 4, 5, 6, 7, 8])
    return Graph.from_edges(10, src, src.new_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_one_hop_per_round_and_worker_never_respreads():
    e = toy_electorate(lollipop())
    state = WorkerState.empty(10)
    converted_by_round = []
    attempts_by_round = []
    for t in range(4):
        alloc = Allocation((1, 0, 0, 0) if t == 0 else (0, 0, 0, 0))
        out = campaign_round(e, alloc, state, e.preferences, B, [0] * 4, WorkerConfig(), MediaConfig(), SURE,
                             generators(t))
        if t == 0:
            assert out.hires == 1 and out.state.worker_ids[0].tolist() == [0] and out.spreaders == 1
        state = out.state
        converted_by_round.append(set(torch.nonzero(state.spreaders).flatten().tolist()))
        attempts_by_round.append(state.attempts.tolist())
    assert converted_by_round[0] == {1, 2, 3}           # the worker's friends, in its purchase round
    assert converted_by_round[1] == {4}                 # converts of round 1 spread in round 2 (0 is a worker)
    assert converted_by_round[2] == {3, 5}              # one hop further each round (3 is an ordinary voter)
    assert 6 in converted_by_round[3]
    # the worker spread only in round 1: 1 and 2 (friends of the worker only) were tried exactly once
    assert attempts_by_round[-1][1] == attempts_by_round[-1][2] == 1
    assert attempts_by_round[-1][0] == 0                 # workers are never targets
    assert bool(state.pinned[0]) and int(state.pinned.sum()) == 1


def test_forfeit_round_still_spreads_earlier_converts():
    e = toy_electorate(lollipop())
    first = campaign_round(e, Allocation((1, 0, 0, 0)), WorkerState.empty(10), e.preferences, B, [0] * 4,
                           WorkerConfig(), MediaConfig(), SURE, generators(0))
    nothing = campaign_round(e, Allocation(), first.state, e.preferences, B, [0] * 4, WorkerConfig(), MediaConfig(),
                             SURE, generators(1))
    assert nothing.cost == Decimal("0.00") and nothing.hires == 0 and nothing.spreaders == 3 and nothing.converted == 1


# ---------------------------------------------------------------------------------- cap (R27)

def test_attempt_cap_counts_failures_and_skips_workers_and_zealots():
    # K_{5,3}: voters 0..4 spread to 5, 6, 7; 6 is a zealot, 7 is blocked as a worker
    src = torch.arange(5).repeat_interleave(3)
    dst = torch.tensor([5, 6, 7]).repeat(5)
    graph = Graph.from_edges(8, src, dst)
    spreaders = torch.zeros(8, dtype=torch.bool)
    spreaders[:5] = True
    assert spreader_edge_counts(graph, spreaders).tolist() == [0, 0, 0, 0, 0, 5, 5, 5]
    blocked = torch.zeros(8, dtype=torch.bool)
    blocked[[6, 7]] = True
    attempts = torch.zeros(8, dtype=torch.int64)
    zero = torch.zeros(8, dtype=torch.float64)
    for hop in range(3):
        converted, attempts = ic_hop(graph, spreaders, zero, attempts, blocked, 3, gen(hop))
        assert not converted.any()
        assert attempts.tolist() == [0, 0, 0, 0, 0, 3, 0, 0]   # failures counted, never more than 3
    # once the cap is reached even a certain conversion cannot happen
    converted, _ = ic_hop(graph, spreaders, torch.ones(8, dtype=torch.float64), attempts, blocked, 3, gen(9))
    assert not converted.any()
    converted, att = ic_hop(graph, spreaders, torch.ones(8, dtype=torch.float64), torch.zeros(8, dtype=torch.int64),
                            blocked, 3, gen(9))
    assert converted.tolist() == [False] * 5 + [True, False, False] and att[5] == 3


def test_zealots_and_workers_never_converted_in_campaign_round():
    e = toy_electorate(lollipop(), zealot=(4,))
    state = WorkerState.empty(10)
    for t in range(5):
        out = campaign_round(e, Allocation((1, 0, 0, 0) if t == 0 else (0, 0, 0, 0)), state, e.preferences, B,
                             [0] * 4, WorkerConfig(), MediaConfig(), SURE, generators(t))
        state = out.state
        assert not bool(state.spreaders[4]) and not bool(state.spreaders[0])
        assert int(state.attempts[4]) == 0
    assert not bool(state.spreaders[5:].any())         # the zealot at 4 blocks the path


# ------------------------------------------------------------------------------- chance (R25)

def test_zero_gamma_or_zero_probability_never_converts():
    star = star_graph(500)
    spreaders = torch.zeros(501, dtype=torch.bool)
    spreaders[0] = True
    none = torch.zeros(501, dtype=torch.bool)
    v = B.repeat(501, 1)
    far = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64).repeat(501, 1)
    gamma = torch.ones(501, dtype=torch.float64)
    assert float(conversion_chance(gamma * 0, v, B, SURE.sensitivity, SURE.threshold).max()) == 0.0
    assert float(conversion_chance(gamma, far, B, NEVER.sensitivity, NEVER.threshold).max()) == 0.0
    for chance in (conversion_chance(gamma * 0, v, B, SURE.sensitivity, SURE.threshold),
                   conversion_chance(gamma, far, B, NEVER.sensitivity, NEVER.threshold)):
        for s in range(5):
            converted, _ = ic_hop(star, spreaders, chance, torch.zeros(501, dtype=torch.int64), none, 3, gen(s))
            assert not converted.any()
    # gamma = P = 1 converts every reachable leaf
    full = conversion_chance(gamma, v, B, SURE.sensitivity, SURE.threshold)
    assert float(full.min()) == 1.0
    converted, _ = ic_hop(star, spreaders, full, torch.zeros(501, dtype=torch.int64), none, 3, gen(0))
    assert bool(converted[1:].all()) and not bool(converted[0])


def test_conversion_probability_is_gamma_times_p_and_compounds():
    leaves = 40_000
    star = star_graph(leaves)
    spreaders = torch.zeros(leaves + 1, dtype=torch.bool)
    spreaders[0] = True
    none = torch.zeros(leaves + 1, dtype=torch.bool)
    v = B.repeat(leaves + 1, 1)
    gamma = torch.full((leaves + 1,), 0.6, dtype=torch.float64)
    voting = VotingConfig()
    chance = conversion_chance(gamma, v, B, voting.sensitivity, voting.threshold)
    p = 0.6 / (1 + math.exp(-voting.sensitivity * (1 - voting.threshold)))
    assert torch.allclose(chance, torch.full_like(chance, p))
    converted, _ = ic_hop(star, spreaders, chance, torch.zeros(leaves + 1, dtype=torch.int64), none, 3, gen(4))
    frac = float(converted[1:].double().mean())
    assert abs(frac - p) < 4 * math.sqrt(p * (1 - p) / leaves)
    # two spreaders on every leaf (K_{2,n}): 1 - (1 - gamma P)^2
    n = 20_000
    src = torch.tensor([0, 1]).repeat_interleave(n)
    dst = torch.arange(2, n + 2).repeat(2)
    g2 = Graph.from_edges(n + 2, src, dst)
    sp = torch.zeros(n + 2, dtype=torch.bool)
    sp[:2] = True
    c2 = torch.full((n + 2,), 0.3, dtype=torch.float64)
    converted, att = ic_hop(g2, sp, c2, torch.zeros(n + 2, dtype=torch.int64), torch.zeros(n + 2, dtype=torch.bool), 3, gen(5))
    q = 1 - 0.7 ** 2
    assert abs(float(converted[2:].double().mean()) - q) < 4 * math.sqrt(q * (1 - q) / n)
    assert bool((att[2:] == 2).all())


def test_ic_hop_batched_matches_single():
    e = electorate(n=2000)
    gamma = e.susceptibility.campaign
    chance = (gamma * 0.5).clamp(0, 1)
    sp = torch.zeros((3, e.num_voters), dtype=torch.bool)
    sp[0, :20] = True
    sp[2, 100:150] = True
    att = torch.zeros((3, e.num_voters), dtype=torch.int64)
    blocked = e.susceptibility.zealot
    batched, batt = ic_hop(e.graph, sp, chance, att, blocked, 3, gen(7))
    for c in range(3):
        single, satt = ic_hop(e.graph, sp[c], chance, att[c], blocked, 3, gen(7))
        assert torch.equal(single, batched[c]) and torch.equal(satt, batt[c])
    assert not batched[1].any()


# ------------------------------------------------------------------------------- pinned (R23)

def test_step_preferences_holds_pinned_workers_exactly_at_b():
    e = electorate(n=2000)
    v = e.preferences.clone()
    zealot = e.susceptibility.zealot
    pinned = torch.zeros(e.num_voters, dtype=torch.bool)
    pinned[torch.nonzero(~zealot).flatten()[:50]] = True
    b = B.to(v.dtype)
    v[pinned] = b
    agg = torch.full_like(v, 0.2)
    media = torch.full((1, e.num_voters), 0.3, dtype=v.dtype)
    new = step_preferences(v, agg, torch.full((e.num_voters,), 0.5, dtype=v.dtype), media, b.unsqueeze(0), zealot,
                           pinned=pinned.unsqueeze(0))
    assert torch.equal(new[pinned], b.expand(int(pinned.sum()), -1))
    assert torch.equal(new[zealot], v[zealot])
    free = ~pinned & ~zealot
    assert not torch.allclose(new[free], v[free])
    with pytest.raises(ValueError):
        step_preferences(v, agg, torch.full((e.num_voters,), 0.5, dtype=v.dtype), media, b.unsqueeze(0), zealot,
                         pinned=pinned)


def test_round_keeps_workers_at_b_and_zealots_unchanged():
    e = electorate(n=3000)
    cfg = GameConfig(seed=3, device="cpu", electorate=ElectorateConfig(num_voters=3000))
    b = B.to(e.preferences.dtype)
    v = e.preferences
    state = WorkerState.empty(e.num_voters)
    for t in range(3):
        alloc = Allocation((2, 1, 1, 1) if t == 0 else (1, 0, 0, 0), (True, True, False, False))
        out = campaign_round(e, alloc, state, v, b, [t, t, 0, 0], cfg.workers, cfg.media, cfg.voting, generators(t))
        assert torch.equal(out.assigned & e.susceptibility.zealot, torch.zeros_like(out.assigned))
        new_v = resolve_campaign_round(e, cfg, v, b.unsqueeze(0), [out])
        state = out.state
        assert torch.equal(new_v[state.pinned], b.expand(int(state.pinned.sum()), -1))
        z = e.susceptibility.zealot
        assert torch.equal(new_v[z], e.preferences[z])
        v = new_v
    assert state.team_by_region == (4, 1, 1, 1)


# ------------------------------------------------------------------------ campaign_round + determinism

ALLOC = Allocation((4, 4, 2, 2), (True, False, True, False))


def test_campaign_round_end_to_end_and_determinism():
    e = electorate()
    cfg = GameConfig()
    b = B.to(e.preferences.dtype)
    run = lambda s: campaign_round(e, ALLOC, WorkerState.empty(e.num_voters), e.preferences, b, [0, 0, 1, 0],  # noqa: E731
                                   cfg.workers, cfg.media, cfg.voting, generators(s))
    out = run(1)
    c = out.intensity
    assert c.shape == (e.num_voters,) and torch.all((c >= 0) & (c <= 1))
    assert not c[e.susceptibility.zealot].any()
    assert out.hires == 12 and out.spreaders == 12 and out.state.team_by_region == (4, 4, 2, 2)
    assert out.converted == int(out.state.spreaders.sum()) and out.media_exposed > 0
    assert out.cost == worker_cost(12, cfg.workers.worker_cost) + media_cost(e.sizes, ALLOC.media_regions, cfg.media.base_cost)
    again = run(1)
    assert torch.equal(c, again.intensity) and torch.equal(out.assigned, again.assigned)
    assert torch.equal(out.state.attempts, again.state.attempts)
    assert not torch.equal(out.assigned, run(2).assigned)


def test_campaign_round_empty_allocation_and_errors():
    e = electorate()
    cfg = GameConfig()
    b = B.to(e.preferences.dtype)
    out = campaign_round(e, Allocation(), WorkerState.empty(e.num_voters), e.preferences, b, [0] * 4, cfg.workers,
                         cfg.media, cfg.voting, generators(1))
    assert not out.intensity.any() and out.cost == Decimal("0.00")
    assert out.hires == out.spreaders == out.converted == out.media_exposed == 0
    with pytest.raises(ValueError):
        campaign_round(e, Allocation((10_000, 0, 0, 0)), WorkerState.empty(e.num_voters), e.preferences, b, [0] * 4,
                       cfg.workers, cfg.media, cfg.voting, generators(1))
    with pytest.raises(ValueError):
        campaign_round(e, ALLOC, WorkerState.empty(e.num_voters), e.preferences, b, [0] * 4, cfg.workers, cfg.media,
                       cfg.voting, {"workers": gen(1)})


@pytest.mark.parametrize("alloc", [
    Allocation((-1, 0, 0, 0)), Allocation((1, 0, 0)), Allocation((1.5, 0, 0, 0)),          # type: ignore[arg-type]
    Allocation((1, 0, 0, 0), (False,) * 3), Allocation((True, 0, 0, 0)),                    # type: ignore[arg-type]
])
def test_allocation_validate_errors(alloc):
    with pytest.raises(ValueError):
        alloc.validate()


def test_allocation_validate_ok():
    Allocation((5, 5, 2, 0), (True, False, False, True)).validate()
    Allocation().validate()
    assert Allocation((1, 2, 3, 4)).total_hires == 10


def test_simulate_world_deterministic_per_seed():
    e = electorate(n=3000)
    cfg = GameConfig(seed=3, device="cpu", electorate=ElectorateConfig(num_voters=3000))
    policy = lambda ctx: Allocation((2, 2, 0, 0) if ctx.round == 0 else (0, 0, 0, 0))  # noqa: E731
    a = simulate_world(e, cfg, B, policy)
    b = simulate_world(e, cfg, B, policy)
    c = simulate_world(e, dataclasses.replace(cfg, seed=4), B, policy)
    assert a.converted == b.converted and a.share_trajectory == b.share_trajectory
    assert a.converted != c.converted
    assert a.spreaders[0] == 4 and a.spreaders[1] == a.converted[0]
    assert a.spent == Decimal(4) * cfg.workers.worker_cost


# -------------------------------------------------------------------------------------- media

def test_media_exposure_fraction():
    region_of = torch.repeat_interleave(torch.arange(4), torch.tensor([20_000, 10_000, 20_000, 10_000]))
    exposed = media_exposure(region_of, [True, False, True, False], 0.6, gen(3))
    for r, on in enumerate([True, False, True, False]):
        frac = float(exposed[region_of == r].double().mean())
        if on:
            assert abs(frac - 0.6) < 4 * math.sqrt(0.24 / 20_000)
        else:
            assert frac == 0.0


def test_media_intensity_decay():
    region_of = torch.repeat_interleave(torch.arange(4), 50)
    beta = torch.full((200,), 0.8, dtype=torch.float64)
    out = media_intensity(region_of, [True, True, True, False], [0, 1, 4, 0], beta, 1.0, 0.15, 0.5, gen(1))
    for r, n in enumerate([0, 1, 4]):
        assert torch.allclose(out[region_of == r], torch.full((50,), 0.15 / (1 + 0.5 * n) * 0.8, dtype=torch.float64))
    assert not out[region_of == 3].any()
    # Same generator seed => same exposure pattern as media_exposure.
    exposed = media_exposure(region_of, [True] * 4, 0.5, gen(7))
    out = media_intensity(region_of, [True] * 4, [0] * 4, beta, 0.5, 0.15, 0.5, gen(7))
    assert torch.equal(out > 0, exposed)


def test_media_validation():
    beta = torch.ones(1000)
    with pytest.raises(ValueError):
        media_intensity(REGION_OF, [True, False], [0] * 4, beta, 0.6, 0.15, 0.5, gen(1))
    with pytest.raises(ValueError):
        media_intensity(REGION_OF, [True] * 4, [0, -1, 0, 0], beta, 0.6, 0.15, 0.5, gen(1))


def test_media_cost_exact_decimal_rounding():
    # N = 3: region shares n_r/(N/4) = 4 n_r / 3.
    assert media_cost([1, 1, 1, 0], [True, False, False, False], Decimal("20.00")) == Decimal("26.67")  # 26.666..
    assert media_cost([1, 1, 1, 0], [True, True, True, True], Decimal("20.00")) == Decimal("80.01")
    # Exact half-cent tie rounds up: 0.01 * 4 * 1 / 8 = 0.005 -> 0.01.
    assert media_cost([1, 7, 0, 0], [True, False, False, False], Decimal("0.01")) == Decimal("0.01")
    assert media_cost(torch.tensor([2500, 2500, 2500, 2500]), [True, False, True, False], Decimal("20.00")) == Decimal("40.00")
    assert media_cost([2500, 3000, 2000, 2500], [False, True, True, False], Decimal("20.00")) == Decimal("40.00")
    assert media_cost([2500, 3001, 2000, 2499], [False, True, False, False], Decimal("20.00")) == Decimal("24.01")  # 24.008
    assert media_cost([2500] * 4, [False] * 4, Decimal("20.00")) == Decimal("0.00")
    with pytest.raises(ValueError):
        media_cost([0, 0, 0, 0], [True] * 4, Decimal("20.00"))
    with pytest.raises(ValueError):
        media_cost([1, 1, 1, 1], [True] * 4, 20.0)


def test_worker_cost():
    assert worker_cost(3, Decimal("25.00")) == Decimal("75.00")
    assert worker_cost(0, Decimal("25.00")) == Decimal("0.00")
    with pytest.raises(ValueError):
        worker_cost(-1, Decimal("25.00"))


# ------------------------------------------------------------------------------------ combine

def test_combine_intensities():
    a = torch.tensor([0.5, 0.0, 1.0, 0.2], dtype=torch.float64)
    b = torch.tensor([0.5, 0.3, 0.4, 0.2], dtype=torch.float64)
    c = torch.tensor([0.0, 0.1, 0.0, 0.2], dtype=torch.float64)
    out = combine_intensities(a, b, c)
    expected = [0.75, 1 - 0.7 * 0.9, 1.0, 1 - 0.8 ** 3]
    assert torch.allclose(out, torch.tensor(expected, dtype=torch.float64))
    frozen = torch.tensor([False, True, False, True])
    out = combine_intensities(a, b, c, frozen=frozen)
    assert out[1] == 0 and out[3] == 0 and math.isclose(float(out[0]), 0.75)
    with pytest.raises(ValueError):
        combine_intensities()


# ------------------------------------------------------------------------------------------ GPU

def _cuda2_free() -> bool:
    if torch.cuda.device_count() <= 2:
        return False
    try:
        return torch.cuda.mem_get_info(2)[0] > 2 * 2**30
    except RuntimeError:
        return False


@pytest.mark.gpu
def test_generator_device_mismatch_rejected():
    if not _cuda2_free():
        pytest.skip("cuda:2 not present or busy")
    star = star_graph(3)
    with pytest.raises(ValueError):
        ic_hop(star, torch.zeros(4, dtype=torch.bool), torch.zeros(4), torch.zeros(4, dtype=torch.int64),
               torch.zeros(4, dtype=torch.bool), 3, gen(1, "cuda:2"))


@pytest.mark.gpu
def test_campaign_on_gpu_matches_cpu_in_distribution():
    if not _cuda2_free():
        pytest.skip("cuda:2 not present or busy")
    device = "cuda:2"
    cfg = GameConfig(seed=3, device="cpu", electorate=ElectorateConfig(num_voters=5000))
    cpu, gpu = electorate(), electorate(device)
    policy = lambda ctx: Allocation((2, 1, 0, 0) if ctx.round == 0 else (0, 0, 0, 0), (ctx.round == 0, False, False, False))  # noqa: E731
    cpu_conv, gpu_conv, cpu_share, gpu_share = [], [], [], []
    for s in range(6):
        c = dataclasses.replace(cfg, seed=s)
        a = simulate_world(cpu, c, B, policy)
        g = simulate_world(gpu, c, B, policy)
        assert g.final_preferences.device == torch.device(device) and a.spent == g.spent
        cpu_conv.append(sum(a.converted))
        gpu_conv.append(sum(g.converted))
        cpu_share.append(a.share_trajectory[-1])
        gpu_share.append(g.share_trajectory[-1])
    assert abs(sum(cpu_share) - sum(gpu_share)) / 6 < 0.02, (cpu_share, gpu_share)
    assert abs(sum(cpu_conv) - sum(gpu_conv)) < 0.5 * max(sum(cpu_conv), 1), (cpu_conv, gpu_conv)
