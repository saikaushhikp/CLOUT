import math

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams
from dumbocracy.electorate import Graph, generate_electorate
from dumbocracy.electorate.regions import minimum_region_size, region_of, sample_region_sizes
from dumbocracy.electorate.sbm import (
    EXACT_PAIR_LIMIT, _triangle_unflatten, block_edge_counts, block_probability_matrix,
    sample_block_probabilities, sample_sbm,
)
from dumbocracy.sectors import archetype_means, personality_beta_params


def make(n=10_000, seed=1, **kw):
    return generate_electorate(GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=n, **kw)))


@pytest.mark.parametrize("n", [4, 20, 100, 1000, 123_457])
def test_region_sizes(n):
    sizes = sample_region_sizes(n, 2.0, torch.Generator().manual_seed(n))
    assert int(sizes.sum()) == n
    assert int(sizes.min()) >= minimum_region_size(n) >= 0.15 * n
    assert torch.equal(sizes, sample_region_sizes(n, 2.0, torch.Generator().manual_seed(n)))


def test_region_sizes_too_small():
    with pytest.raises(ValueError):
        sample_region_sizes(7, 2.0, torch.Generator(), 0.15)


def test_triangle_unflatten_is_a_bijection():
    for n in (2, 3, 7, 50):
        flat = torch.arange(n * (n - 1) // 2)
        i, j = _triangle_unflatten(flat, n)
        assert torch.all((0 <= i) & (i < j) & (j < n))
        assert len({(a, b) for a, b in zip(i.tolist(), j.tolist())}) == len(flat)


def test_graph_from_edges_canonicalises():
    g = Graph.from_edges(4, torch.tensor([1, 0, 2, 2, 3, 1]), torch.tensor([0, 1, 2, 3, 2, 3]))
    assert g.src.tolist() == [0, 1, 2] and g.dst.tolist() == [1, 3, 3]
    assert g.degree.tolist() == [1, 2, 1, 2]
    assert g.indptr.tolist() == [0, 1, 3, 4, 6]
    assert g.indices.tolist() == [1, 0, 3, 3, 1, 2]


def test_block_probabilities_and_sparsity_factor():
    p_r, p_rs = sample_block_probabilities((0.5, 0.9), (0.01, 0.2), torch.Generator().manual_seed(3))
    assert torch.all((p_r >= 0.5) & (p_r <= 0.9))
    off = p_rs[~torch.eye(4, dtype=torch.bool)]
    assert torch.all((off >= 0.01) & (off <= 0.2)) and torch.equal(p_rs, p_rs.T)
    sizes = torch.tensor([2500, 2500, 2500, 2500])
    P, s, capped = block_probability_matrix(sizes, p_r, p_rs, 16.0)
    assert not capped and torch.allclose(P.diag(), s * p_r)
    n = sizes.double()
    expected_degree = (P.diag() * n * (n - 1)).sum() + sum(P[r, t] * n[r] * n[t] for r in range(4) for t in range(4) if r != t)
    assert math.isclose(float(expected_degree) / 10_000, 16.0, rel_tol=1e-9)
    # Tiny graph: the cap keeps every probability <= 1.
    P, s, capped = block_probability_matrix(torch.tensor([1, 1, 1, 1]), p_r, p_rs, 16.0)
    assert capped and float(P.max()) <= 1.0 + 1e-12


@pytest.mark.parametrize("n", [2_000, 60_000])  # exact-mask path and Binomial path
def test_sbm_block_counts_match_binomial(n):
    sizes = torch.tensor([n // 4] * 4)
    p_r, p_rs = sample_block_probabilities((0.5, 0.9), (0.01, 0.2), torch.Generator().manual_seed(5))
    P, _, _ = block_probability_matrix(sizes, p_r, p_rs, 16.0)
    if n == 60_000:
        assert (n // 4) * (n // 4) > EXACT_PAIR_LIMIT // 100  # large blocks exercise the Binomial path
    g = sample_sbm(sizes, P, torch.Generator().manual_seed(7))
    assert torch.all(g.src < g.dst)
    assert len(torch.unique(g.src * n + g.dst)) == g.num_edges
    counts = block_edge_counts(g, region_of(sizes)).double()
    for r in range(4):
        for t in range(r, 4):
            m = n // 4
            pairs = m * (m - 1) / 2 if r == t else m * m
            mean = pairs * float(P[r, t])
            assert abs(float(counts[r, t]) - mean) <= 5 * math.sqrt(mean) + 2, (r, t)
    assert abs(2 * g.num_edges / n - 16.0) < 0.6


def test_default_mean_degree_is_30():
    e = make(n=20_000)
    assert abs(2 * e.graph.num_edges / e.num_voters - 30.0) < 1.0


def test_electorate_invariants_and_census_privacy():
    e = make()
    n = e.num_voters
    v = e.preferences
    assert v.shape == (n, 5) and torch.all(v >= 0)
    assert float((v.sum(1) - 1).abs().max()) < 1e-5
    assert sorted(e.archetype_of_region.tolist()) == sorted(set(e.archetype_of_region.tolist()))
    assert sorted(e.personality_of_region.tolist()) == [0, 1, 2, 3]
    sus = e.susceptibility
    z = sus.zealot
    assert abs(float(z.double().mean()) - 0.05) < 0.01
    for x in (sus.social, sus.media, sus.campaign):
        assert torch.all((x >= 0) & (x <= 1)) and torch.all(x[z] == 0)
    census = e.census()
    text = repr(census)
    for forbidden in ("archetype", "personality", "p_r", "sparsity", "preferences", "zealot"):
        assert forbidden not in text
    assert sum(r["voters"] for r in census["regions"]) == n
    assert sum(r["intra_edges"] for r in census["regions"]) + sum(p["edges"] for p in census["inter_edges"]) == e.graph.num_edges


def test_region_preference_means_follow_archetypes():
    e = make(n=40_000, seed=11)
    means = archetype_means(dtype=torch.float32)
    for r in range(4):
        empirical = e.preferences[e.region_of == r].mean(0)
        assert torch.allclose(empirical, means[e.archetype_of_region[r]], atol=0.01)


def test_susceptibility_means_follow_personalities():
    e = make(n=40_000, seed=12, zealot_fraction=0.0)
    params = personality_beta_params(dtype=torch.float64)
    for r in range(4):
        a, b = params[e.personality_of_region[r], 0]
        empirical = float(e.susceptibility.social[e.region_of == r].double().mean())
        assert abs(empirical - float(a / (a + b))) < 0.01


def test_seed_determinism_and_sensitivity():
    a, b, c = make(seed=4), make(seed=4), make(seed=5)
    assert torch.equal(a.graph.src, b.graph.src) and torch.equal(a.preferences, b.preferences)
    assert not torch.equal(a.preferences, c.preferences)


@pytest.mark.gpu
def test_canonical_generation_is_bit_identical_across_devices():
    cfg = GameConfig(seed=21, electorate=ElectorateConfig(num_voters=20_000))
    cpu = generate_electorate(cfg, device="cpu")
    gpu = generate_electorate(cfg, device="cuda")
    assert gpu.preferences.is_cuda and gpu.graph.src.is_cuda
    assert torch.equal(cpu.graph.src, gpu.graph.src.cpu()) and torch.equal(cpu.graph.dst, gpu.graph.dst.cpu())
    assert torch.equal(cpu.preferences, gpu.preferences.cpu())
    assert torch.equal(cpu.susceptibility.zealot, gpu.susceptibility.zealot.cpu())


@pytest.mark.gpu
def test_device_generation_agrees_in_distribution():
    cfg = GameConfig(seed=21, generation="device", electorate=ElectorateConfig(num_voters=20_000))
    cpu = generate_electorate(cfg, device="cpu")
    gpu = generate_electorate(cfg, device="cuda")
    assert torch.equal(cpu.sizes, gpu.sizes)
    assert torch.equal(cpu.archetype_of_region, gpu.archetype_of_region)
    assert torch.equal(cpu.personality_of_region, gpu.personality_of_region)
    assert torch.allclose(cpu.edge_probability, gpu.edge_probability)
    assert abs(cpu.graph.num_edges - gpu.graph.num_edges) < 5 * math.sqrt(cpu.graph.num_edges)
    assert torch.allclose(cpu.preferences.mean(0), gpu.preferences.cpu().mean(0), atol=3e-3)
