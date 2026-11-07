import itertools
import math

import pytest
import torch

from dumbocracy.analysis.robustness import (
    exact_robustness, outside_neighbour_counts, reachability, robustness_upper_bound, rs_upper_bound,
)
from dumbocracy.analysis.spectral import (
    algebraic_connectivity, connected_components, degroot_spectral_gap, induced_subgraph,
    largest_eigenpair, region_lambda_max,
)
from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.electorate import Graph, generate_electorate


def complete(n):
    pairs = list(itertools.combinations(range(n), 2))
    return Graph.from_edges(n, torch.tensor([a for a, _ in pairs]), torch.tensor([b for _, b in pairs]))


def path(n):
    return Graph.from_edges(n, torch.arange(n - 1), torch.arange(1, n))


def test_lambda2_known_graphs():
    assert math.isclose(algebraic_connectivity(complete(6)), 6.0, rel_tol=1e-9)
    assert math.isclose(algebraic_connectivity(path(10)), 2 - 2 * math.cos(math.pi / 10), rel_tol=1e-9)
    two_parts = Graph.from_edges(4, torch.tensor([0, 2]), torch.tensor([1, 3]))
    assert algebraic_connectivity(two_parts) < 1e-9 and connected_components(two_parts) == 2
    big_path = path(600)  # Lanczos branch
    assert math.isclose(algebraic_connectivity(big_path), 2 - 2 * math.cos(math.pi / 600), rel_tol=1e-3)


def test_lambda_max_and_centrality():
    value, vec = largest_eigenpair(complete(5))
    assert math.isclose(value, 4.0, rel_tol=1e-6)
    assert torch.allclose(vec, torch.full((5,), 5 ** -0.5, dtype=torch.float64), atol=1e-6)
    star = Graph.from_edges(5, torch.zeros(4, dtype=torch.long), torch.arange(1, 5))
    value, vec = largest_eigenpair(star)
    assert math.isclose(value, 2.0, rel_tol=1e-6) and int(torch.argmax(vec)) == 0


def test_degroot_gap_complete_graph():
    # D^-1 A for K_n has eigenvalues 1 and -1/(n-1).
    assert math.isclose(degroot_spectral_gap(complete(5)), 1 - 1 / 4, rel_tol=1e-9)


def test_robustness_exact_small_graphs():
    assert exact_robustness(complete(4)) == 2
    assert exact_robustness(complete(5)) == 3
    assert exact_robustness(path(5)) == 1
    disconnected = Graph.from_edges(4, torch.tensor([0, 2]), torch.tensor([1, 3]))
    assert exact_robustness(disconnected) == 0


def test_reachability_counts():
    g = path(4)  # 0-1-2-3
    s = torch.tensor([True, True, False, False])
    assert outside_neighbour_counts(g, s).tolist() == [0, 1, 0, 0]
    assert reachability(g, s) == 1


def test_upper_bound_is_valid_on_small_graphs():
    region_of = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(28, generator=gen) < 0.6
        pairs = torch.tensor(list(itertools.combinations(range(8), 2)))[mask]
        g = Graph.from_edges(8, pairs[:, 0], pairs[:, 1])
        assert robustness_upper_bound(g, region_of).r_upper >= exact_robustness(g)


def test_electorate_analysis_smoke():
    e = generate_electorate(GameConfig(seed=2, device="cpu", electorate=ElectorateConfig(num_voters=3000)))
    bound = robustness_upper_bound(e.graph, e.region_of)
    assert 0 <= bound.r_upper <= bound.min_degree
    assert rs_upper_bound(e.graph, e.region_of, 1) >= 0
    lam = region_lambda_max(e.graph, e.region_of)
    assert all(v > 0 for v in lam)
    sub = induced_subgraph(e.graph, e.region_of == 0)
    assert sub.num_nodes == int(e.sizes[0])
    assert 0 < degroot_spectral_gap(e.graph) <= 1 or connected_components(e.graph) > 1
