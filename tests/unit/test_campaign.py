import math
from decimal import Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig, MediaConfig, WorkerConfig
from dumbocracy.dynamics.campaign import (
    Allocation, CascadeResult, campaign_round, combine_intensities, media_cost, media_exposure,
    media_intensity, sir_cascade, word_of_mouth_intensity, worker_contacts, worker_cost, worker_intensity,
)
from dumbocracy.electorate import Graph, generate_electorate


def gen(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def path_graph(n):
    return Graph.from_edges(n, torch.arange(n - 1), torch.arange(1, n))


def ring_graph(n):
    return Graph.from_edges(n, torch.arange(n), (torch.arange(n) + 1) % n)


def star_graph(leaves):
    return Graph.from_edges(leaves + 1, torch.zeros(leaves, dtype=torch.int64), torch.arange(1, leaves + 1))


def mask(n, *idx):
    out = torch.zeros(n, dtype=torch.bool)
    out[list(idx)] = True
    return out


REGION_OF = torch.repeat_interleave(torch.arange(4), torch.tensor([100, 200, 300, 400]))


# ------------------------------------------------------------------------------------- workers

def test_worker_contacts_totals_and_regions():
    counts = worker_contacts(REGION_OF, [3, 0, 5, 0], 7, gen(1))
    assert counts.dtype == torch.int64 and counts.shape == (1000,)
    assert int(counts.sum()) == (3 + 5) * 7
    assert int(counts[REGION_OF == 0].sum()) == 21 and int(counts[REGION_OF == 2].sum()) == 35
    assert int(counts[(REGION_OF == 1) | (REGION_OF == 3)].sum()) == 0


def test_worker_contacts_reproducible_and_seed_sensitive():
    a = worker_contacts(REGION_OF, [10, 10, 10, 10], 20, gen(5))
    b = worker_contacts(REGION_OF, [10, 10, 10, 10], 20, gen(5))
    c = worker_contacts(REGION_OF, [10, 10, 10, 10], 20, gen(6))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_worker_contacts_non_contiguous_regions():
    shuffled = REGION_OF[torch.randperm(1000, generator=gen(0))]
    counts = worker_contacts(shuffled, [0, 4, 0, 0], 10, gen(2))
    assert int(counts[shuffled == 1].sum()) == 40 and int(counts.sum()) == 40


def test_worker_contacts_zero_and_validation():
    assert torch.equal(worker_contacts(REGION_OF, [0, 0, 0, 0], 9, gen(1)), torch.zeros(1000, dtype=torch.int64))
    assert int(worker_contacts(REGION_OF, [5, 5, 5, 5], 0, gen(1)).sum()) == 0
    for bad in ([1, 2, 3], [-1, 0, 0, 0], [1.5, 0, 0, 0], [True, 0, 0, 0]):
        with pytest.raises(ValueError):
            worker_contacts(REGION_OF, bad, 3, gen(1))
    with pytest.raises(ValueError):
        worker_contacts(REGION_OF, [1, 0, 0, 0], -1, gen(1))


def test_worker_intensity_compounds():
    counts = torch.tensor([0, 1, 2, 3])
    gamma = torch.tensor([0.5, 0.5, 0.5, 1.0], dtype=torch.float64)
    out = worker_intensity(counts, gamma, 0.3)
    expected = [0.0, 0.15, 1 - 0.85 ** 2, 1 - 0.7 ** 3]
    assert torch.allclose(out, torch.tensor(expected, dtype=torch.float64))


# ---------------------------------------------------------------------------------------- SIR

def test_sir_no_transmission_reaches_nobody():
    g = ring_graph(20)
    res = sir_cascade(g, mask(20, 0), torch.ones(20, dtype=torch.bool), 0.0, 0.0, 10, gen(1))
    assert isinstance(res, CascadeResult)
    assert not res.reached.any() and res.per_hop_new == [0] * 10


@pytest.mark.parametrize("graph,n,hops", [(path_graph(12), 12, 11), (ring_graph(15), 15, 7)])
def test_sir_full_reach_of_component(graph, n, hops):
    res = sir_cascade(graph, mask(n, 0), torch.ones(n, dtype=torch.bool), 1.0, 0.0, hops, gen(3))
    assert torch.equal(res.reached, ~mask(n, 0))
    assert sum(res.per_hop_new) == n - 1


def test_sir_component_bounded():
    # Two disjoint paths 0-1-2 and 3-4-5: seeding 0 never reaches the other component.
    g = Graph.from_edges(6, torch.tensor([0, 1, 3, 4]), torch.tensor([1, 2, 4, 5]))
    res = sir_cascade(g, mask(6, 0), torch.ones(6, dtype=torch.bool), 1.0, 0.0, 10, gen(1))
    assert res.reached.tolist() == [False, True, True, False, False, False]


def test_sir_hop_limit_on_path():
    res = sir_cascade(path_graph(10), mask(10, 0), torch.ones(10, dtype=torch.bool), 1.0, 0.0, 2, gen(1))
    assert res.reached.nonzero().squeeze(1).tolist() == [1, 2] and res.per_hop_new == [1, 1]


def test_sir_zealots_block_and_are_never_reached():
    n = 10
    can_spread = torch.ones(n, dtype=torch.bool)
    can_spread[4] = False
    res = sir_cascade(path_graph(n), mask(n, 0), can_spread, 1.0, 0.0, 20, gen(1))
    assert res.reached.nonzero().squeeze(1).tolist() == [1, 2, 3]
    # A zealot seed does not spread either.
    can_spread = torch.ones(n, dtype=torch.bool)
    can_spread[0] = False
    res = sir_cascade(path_graph(n), mask(n, 0), can_spread, 1.0, 0.0, 20, gen(1))
    assert not res.reached.any()


def test_sir_seeds_excluded_and_recovery_one_still_transmits_once():
    # Seeds 0 and 2 on a path: node 1 is reached once; seed 2 is never "reached".
    res = sir_cascade(path_graph(5), mask(5, 0, 2), torch.ones(5, dtype=torch.bool), 1.0, 1.0, 5, gen(1))
    # Seeds transmit at hop 1 then recover; newly reached 1 and 3 transmit at hop 2 (reaching 4).
    assert res.reached.tolist() == [False, True, False, True, True]
    assert res.per_hop_new == [2, 1, 0, 0, 0]


def test_sir_per_hop_sums_and_determinism():
    g = ring_graph(200)
    seeds = mask(200, 0, 50, 100)
    can = torch.ones(200, dtype=torch.bool)
    a = sir_cascade(g, seeds, can, 0.7, 0.3, 6, gen(9))
    b = sir_cascade(g, seeds, can, 0.7, 0.3, 6, gen(9))
    assert torch.equal(a.reached, b.reached) and a.per_hop_new == b.per_hop_new
    assert sum(a.per_hop_new) == int(a.reached.sum()) and len(a.per_hop_new) == 6
    assert not (a.reached & seeds).any()


def test_sir_star_reach_fraction():
    leaves, p = 4000, 0.3
    res = sir_cascade(star_graph(leaves), mask(leaves + 1, 0), torch.ones(leaves + 1, dtype=torch.bool), p, 0.5, 1, gen(4))
    frac = int(res.reached.sum()) / leaves
    assert abs(frac - p) < 4 * math.sqrt(p * (1 - p) / leaves)


def test_sir_validation():
    g = path_graph(4)
    ones = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError):
        sir_cascade(g, ones, ones, 1.5, 0.0, 1, gen(1))
    with pytest.raises(ValueError):
        sir_cascade(g, ones, ones, 0.5, 0.0, -1, gen(1))
    with pytest.raises(ValueError):
        sir_cascade(g, torch.ones(3, dtype=torch.bool), ones, 0.5, 0.0, 1, gen(1))


def test_sir_reach_grows_across_spectral_threshold():
    """ER graph, N = 5000, k̄ ≈ 30 → λ_max ≈ 31, threshold ρ/λ_max ≈ 0.016 at ρ = 0.5."""
    n, k = 5000, 30
    g0 = gen(11)
    m = n * k // 2
    g = Graph.from_edges(n, torch.randint(0, n, (m,), generator=g0), torch.randint(0, n, (m,), generator=g0))
    can = torch.ones(n, dtype=torch.bool)
    num_seeds = 20
    means = []
    for tau in (0.005, 0.02, 0.05, 0.1):
        reach = []
        for s in range(8):
            seeds = mask(n, *torch.randperm(n, generator=gen(100 + s))[:num_seeds].tolist())
            reach.append(int(sir_cascade(g, seeds, can, tau, 0.5, 6, gen(200 + s)).reached.sum()))
        means.append(sum(reach) / len(reach))
    assert all(a < b for a, b in zip(means, means[1:])), means
    assert means[0] < 2 * num_seeds and means[-1] > 20 * num_seeds, means


def test_word_of_mouth_intensity():
    reached = torch.tensor([True, False, True])
    gamma = torch.tensor([0.5, 0.9, 0.2], dtype=torch.float64)
    out = word_of_mouth_intensity(reached, gamma, 0.1)
    assert torch.allclose(out, torch.tensor([0.05, 0.0, 0.02], dtype=torch.float64))


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
    assert worker_cost(3, Decimal("4.00")) == Decimal("12.00")
    assert worker_cost(0, Decimal("4.00")) == Decimal("0.00")
    with pytest.raises(ValueError):
        worker_cost(-1, Decimal("4.00"))


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


# --------------------------------------------------------------------------------- allocation

@pytest.mark.parametrize("alloc,free", [
    (Allocation((5, 5, 1, 0), 0, (False,) * 4), 10),
    (Allocation((-1, 0, 0, 0), 0, (False,) * 4), 10),
    (Allocation((1, 0, 0), 0, (False,) * 4), 10),
    (Allocation((1, 0, 0, 0), -2, (False,) * 4), 10),
    (Allocation((1, 0, 0, 0), 0, (False,) * 3), 10),
    (Allocation((8, 4, 0, 0), 1, (False,) * 4), 10),
])
def test_allocation_validate_errors(alloc, free):
    with pytest.raises(ValueError):
        alloc.validate(free)


def test_allocation_validate_ok():
    Allocation((5, 5, 2, 0), 2, (True, False, False, True)).validate(10)
    Allocation((0, 0, 3, 0), 0, (False,) * 4).validate(10)  # unassigned workers are wasted, not an error


# ------------------------------------------------------------------------------- campaign_round

def electorate(device="cpu", seed=3, n=5000):
    return generate_electorate(GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=n)), device=device)


def generators(seed, device="cpu"):
    return {name: gen(seed * 10 + i, device) for i, name in enumerate(("workers", "cascade", "media"))}


ALLOC = Allocation((4, 4, 2, 2), 2, (True, False, True, False))


def test_campaign_round_end_to_end():
    e = electorate()
    out = campaign_round(e, ALLOC, [0, 0, 1, 0], WorkerConfig(), MediaConfig(), generators(1))
    c = out.intensity
    assert c.shape == (e.num_voters,) and c.dtype == e.susceptibility.campaign.dtype
    assert torch.all((c >= 0) & (c <= 1))
    assert not c[e.susceptibility.zealot].any()
    q = WorkerConfig().contacts_per_worker(e.num_voters)
    assert 0 < out.direct_contacts <= 12 * q
    assert out.media_exposed > 0 and out.word_of_mouth_reached >= 0
    expected_cost = Decimal("8.00") + media_cost(e.sizes, ALLOC.media_regions, MediaConfig().base_cost)
    assert out.cost == expected_cost
    again = campaign_round(e, ALLOC, [0, 0, 1, 0], WorkerConfig(), MediaConfig(), generators(1))
    assert torch.equal(c, again.intensity) and out.direct_contacts == again.direct_contacts
    assert out.word_of_mouth_reached == again.word_of_mouth_reached and out.media_exposed == again.media_exposed
    other = campaign_round(e, ALLOC, [0, 0, 1, 0], WorkerConfig(), MediaConfig(), generators(2))
    assert not torch.equal(c, other.intensity)


def test_campaign_round_empty_allocation_and_errors():
    e = electorate()
    out = campaign_round(e, Allocation((0, 0, 0, 0), 0, (False,) * 4), [0] * 4, WorkerConfig(), MediaConfig(), generators(1))
    assert not out.intensity.any() and out.cost == Decimal("0.00")
    assert out.direct_contacts == out.word_of_mouth_reached == out.media_exposed == 0
    with pytest.raises(ValueError):
        campaign_round(e, Allocation((11, 0, 0, 0), 0, (False,) * 4), [0] * 4, WorkerConfig(), MediaConfig(), generators(1))
    with pytest.raises(ValueError):
        campaign_round(e, ALLOC, [0] * 4, WorkerConfig(), MediaConfig(), {"workers": gen(1)})


@pytest.mark.gpu
def test_generator_device_mismatch_rejected():
    if torch.cuda.device_count() <= 2:
        pytest.skip("cuda:2 not present")
    with pytest.raises(ValueError):
        worker_contacts(REGION_OF, [1, 0, 0, 0], 3, gen(1, "cuda:2"))


@pytest.mark.gpu
def test_campaign_round_on_gpu_matches_cpu_in_distribution():
    if torch.cuda.device_count() <= 2:
        pytest.skip("cuda:2 not present")
    device = "cuda:2"
    cpu, gpu = electorate(), electorate(device)
    cpu_means, gpu_means = [], []
    for s in range(6):
        a = campaign_round(cpu, ALLOC, [0] * 4, WorkerConfig(), MediaConfig(), generators(s))
        b = campaign_round(gpu, ALLOC, [0] * 4, WorkerConfig(), MediaConfig(), generators(s, device))
        assert b.intensity.device == torch.device(device)
        assert torch.all((b.intensity >= 0) & (b.intensity <= 1))
        assert not b.intensity[gpu.susceptibility.zealot].any()
        assert a.cost == b.cost
        cpu_means.append(float(a.intensity.mean()))
        gpu_means.append(float(b.intensity.mean()))
    cpu_mean, gpu_mean = sum(cpu_means) / 6, sum(gpu_means) / 6
    assert abs(cpu_mean - gpu_mean) < 0.05 * cpu_mean, (cpu_means, gpu_means)
