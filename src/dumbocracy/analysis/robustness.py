"""Resilient-consensus robustness bounds (spec §11; LeBlanc, Zhang, Koutsoukos & Sundaram 2013).

Definitions (undirected graph, V_i = neighbours of i):
* S is **r-reachable** if ∃ i ∈ S with |V_i \\ S| ≥ r.
* G is **r-robust** if for every pair of nonempty disjoint S1, S2 at least one is r-reachable.
* G is **(r,s)-robust** if for every such pair: all of S1 is r-reachable-by-node, or all of S2 is, or
  |X^r_S1| + |X^r_S2| ≥ s, where X^r_S = {i ∈ S : |V_i \\ S| ≥ r}.
W-MSR with parameter F reaches resilient consensus under F-total malicious nodes iff the graph is
(F+1, F+1)-robust; (2F+1)-robust suffices under the F-local model.

Deciding r-robustness is coNP-complete (r ≥ 2), so for real electorates we report **upper bounds**:
every candidate pair (S1, S2) that fails gives a certificate that the graph is *not* more robust.
Candidates: region unions, minimum-degree singletons, and greedy shrinking of region sets.
``exact_robustness`` brute-forces tiny graphs (n ≤ 10) for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

from ..electorate.sbm import Graph


def outside_neighbour_counts(graph: Graph, in_set: torch.Tensor) -> torch.Tensor:
    """|V_i \\ S| for every node (meaningful for i ∈ S)."""
    rows = graph.rows()
    crossing = in_set[rows] & ~in_set[graph.indices]
    return torch.bincount(rows[crossing], minlength=graph.num_nodes)


def reachability(graph: Graph, in_set: torch.Tensor) -> int:
    """Largest r such that S is r-reachable (max outside-neighbour count over i ∈ S)."""
    counts = outside_neighbour_counts(graph, in_set)
    return int(counts[in_set].max()) if bool(in_set.any()) else 0


def reachable_count(graph: Graph, in_set: torch.Tensor, r: int) -> int:
    """|X^r_S|: nodes of S with at least r neighbours outside S."""
    counts = outside_neighbour_counts(graph, in_set)
    return int((counts[in_set] >= r).sum())


def pair_r_bound(graph: Graph, s1: torch.Tensor, s2: torch.Tensor) -> int:
    """The pair certifies the graph is not (bound+1)-robust: bound = max(reach(S1), reach(S2))."""
    return max(reachability(graph, s1), reachability(graph, s2))


def pair_violates_rs(graph: Graph, s1: torch.Tensor, s2: torch.Tensor, r: int, s: int) -> bool:
    x1, x2 = reachable_count(graph, s1, r), reachable_count(graph, s2, r)
    return x1 < int(s1.sum()) and x2 < int(s2.sum()) and x1 + x2 < s


@dataclass(frozen=True)
class RobustnessBound:
    r_upper: int          # graph is NOT (r_upper + 1)-robust
    witness: str          # human-readable description of the certifying pair
    min_degree: int


def _region_union_pairs(region_of: torch.Tensor, num_regions: int):
    for labels in product((0, 1, 2), repeat=num_regions):  # 0 = neither, 1 = S1, 2 = S2
        if 1 not in labels or 2 not in labels or labels.index(1) > labels.index(2):
            continue
        s1 = torch.zeros_like(region_of, dtype=torch.bool)
        s2 = torch.zeros_like(region_of, dtype=torch.bool)
        for region, label in enumerate(labels):
            if label == 1:
                s1 |= region_of == region
            elif label == 2:
                s2 |= region_of == region
        yield labels, s1, s2


def _greedy_shrink(graph: Graph, start: torch.Tensor, steps: int = 30, drop_fraction: float = 0.05) -> tuple[int, torch.Tensor]:
    """Remove the most outward-connected nodes from S repeatedly; return the best (lowest) reach seen."""
    current = start.clone()
    best_reach, best_set = reachability(graph, current), current.clone()
    for _ in range(steps):
        size = int(current.sum())
        if size <= 1:
            break
        counts = outside_neighbour_counts(graph, current)
        counts = torch.where(current, counts, torch.full_like(counts, -1))
        drop = max(1, int(drop_fraction * size))
        worst = torch.topk(counts, drop).indices
        current[worst] = False
        if not bool(current.any()):
            break
        reach = reachability(graph, current)
        if reach < best_reach:
            best_reach, best_set = reach, current.clone()
    return best_reach, best_set


def robustness_upper_bound(graph: Graph, region_of: torch.Tensor, num_regions: int = 4, greedy: bool = True) -> RobustnessBound:
    """Upper bound on r such that the graph is r-robust, with a certificate description."""
    region_of = region_of.to(graph.device)
    min_degree = int(graph.degree.min()) if graph.num_nodes else 0
    node = int(torch.argmin(graph.degree))
    best = (min_degree, f"singleton {{{node}}} vs any disjoint set (min degree)")
    shrunk: dict[int, tuple[int, torch.Tensor]] = {}
    for labels, s1, s2 in _region_union_pairs(region_of, num_regions):
        bound = pair_r_bound(graph, s1, s2)
        if bound < best[0]:
            best = (bound, f"region union pair {labels}")
    if greedy:
        for region in range(num_regions):
            shrunk[region] = _greedy_shrink(graph, region_of == region)
        for a in range(num_regions):
            for b in range(a + 1, num_regions):
                bound = max(shrunk[a][0], shrunk[b][0])
                if bound < best[0]:
                    best = (bound, f"greedy-shrunk cores of regions {a} and {b}")
    return RobustnessBound(best[0], best[1], min_degree)


def rs_upper_bound(graph: Graph, region_of: torch.Tensor, r: int, num_regions: int = 4) -> int:
    """Upper bound on s for (r, s)-robustness from region-union pairs (n + 1 means no violation found)."""
    region_of = region_of.to(graph.device)
    bound = graph.num_nodes + 1
    for _, s1, s2 in _region_union_pairs(region_of, num_regions):
        x1, x2 = reachable_count(graph, s1, r), reachable_count(graph, s2, r)
        if x1 < int(s1.sum()) and x2 < int(s2.sum()):
            bound = min(bound, x1 + x2)
    return bound


def exact_robustness(graph: Graph) -> int:
    """Exact max r such that the graph is r-robust (brute force; n ≤ 10)."""
    n = graph.num_nodes
    if n > 10:
        raise ValueError("exact_robustness is only for tiny graphs (n <= 10)")
    if n <= 1:
        return 1 if n == 1 else 0
    best = n
    for labels in product((0, 1, 2), repeat=n):
        if 1 not in labels or 2 not in labels or labels.index(1) > labels.index(2):
            continue
        s1 = torch.tensor([label == 1 for label in labels], device=graph.device)
        s2 = torch.tensor([label == 2 for label in labels], device=graph.device)
        best = min(best, pair_r_bound(graph, s1, s2))
    return best
