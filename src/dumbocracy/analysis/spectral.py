"""Spectral diagnostics (spec §11; lectures: consensus, infection models).

* ``algebraic_connectivity``: λ2(L) of the Laplacian — consensus/mixing speed (ẋ = −Lx).
* ``degroot_spectral_gap``: 1 − max(|λ2|, |λ_min|) of the random-walk matrix D⁻¹A — speed of
  the discrete DeGroot averaging used in the dynamics.
* ``largest_eigenpair``: λ_max(A) and the Perron vector (eigenvector centrality) — SIR threshold
  τ·λ_max(A) > ρ and seeding importance. Works on the whole graph or one region block (A_rr).

λ2 computations use SciPy Lanczos on CPU (analysis-only, recommended N ≤ 1e5); λ_max uses GPU power
iteration and scales to 1e6.
"""
from __future__ import annotations

import numpy as np
import torch

from ..electorate.sbm import Graph


def _scipy_adjacency(graph: Graph):
    from scipy.sparse import csr_matrix

    rows = graph.rows().cpu().numpy()
    cols = graph.indices.cpu().numpy()
    data = np.ones(len(cols), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(graph.num_nodes, graph.num_nodes))


def induced_subgraph(graph: Graph, mask: torch.Tensor) -> Graph:
    """Subgraph induced by the boolean node mask, nodes relabelled 0..k-1."""
    mask = mask.to(graph.device)
    new_index = torch.cumsum(mask.to(torch.int64), 0) - 1
    keep = mask[graph.src] & mask[graph.dst]
    return Graph.from_edges(int(mask.sum()), new_index[graph.src[keep]], new_index[graph.dst[keep]])


def connected_components(graph: Graph) -> int:
    from scipy.sparse.csgraph import connected_components as cc

    return int(cc(_scipy_adjacency(graph), directed=False)[0])


def algebraic_connectivity(graph: Graph) -> float:
    """λ2 of L = D − A (0 for a disconnected graph)."""
    n = graph.num_nodes
    if n < 2:
        return 0.0
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    adjacency = _scipy_adjacency(graph)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree) - adjacency
    if n <= 400:
        values = np.linalg.eigvalsh(laplacian.toarray())
        return float(max(values[1], 0.0))
    shift = 2.0 * float(degree.max()) + 1.0  # largest eigenpairs of (shift·I − L) = smallest of L
    values = eigsh(shift * diags(np.ones(n)) - laplacian, k=2, which="LA", return_eigenvectors=False, tol=1e-8)
    return float(max(shift - min(values), 0.0))


def degroot_spectral_gap(graph: Graph) -> float:
    """1 − (second largest |eigenvalue|) of D⁻¹A over non-isolated nodes."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    adjacency = _scipy_adjacency(graph)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    keep = degree > 0
    adjacency = adjacency[keep][:, keep]
    inv_sqrt = diags(1.0 / np.sqrt(degree[keep]))
    symmetric = inv_sqrt @ adjacency @ inv_sqrt  # same spectrum as D⁻¹A
    n = symmetric.shape[0]
    if n <= 400:
        values = np.sort(np.linalg.eigvalsh(symmetric.toarray()))
        top, second, bottom = values[-1], values[-2], values[0]
    else:
        high = eigsh(symmetric, k=2, which="LA", return_eigenvectors=False, tol=1e-8)
        low = eigsh(symmetric, k=1, which="SA", return_eigenvectors=False, tol=1e-8)
        second, bottom = float(min(high)), float(low[0])
    return float(1.0 - max(abs(second), abs(bottom)))


def largest_eigenpair(graph: Graph, iterations: int = 500, tol: float = 1e-9) -> tuple[float, torch.Tensor]:
    """λ_max(A) and the unit Perron vector by power iteration on A + I (on the graph's device)."""
    n = graph.num_nodes
    if graph.num_edges == 0:
        return 0.0, torch.full((n,), 1.0 / max(n, 1) ** 0.5, dtype=torch.float64, device=graph.device)
    rows = graph.rows()
    x = torch.ones(n, dtype=torch.float64, device=graph.device) / n ** 0.5
    value = 0.0
    for _ in range(iterations):
        y = x.clone()
        y.index_add_(0, rows, x[graph.indices])  # (A + I) x
        new_value = float(torch.dot(x, y))
        y = y / torch.linalg.vector_norm(y)
        converged = abs(new_value - value) <= tol * max(1.0, abs(new_value))
        x, value = y, new_value
        if converged:
            break
    return value - 1.0, x.abs()


def region_lambda_max(graph: Graph, region_of: torch.Tensor, num_regions: int = 4) -> list[float]:
    """λ_max(A_rr) of each region's induced subgraph (cascade threshold per region)."""
    return [largest_eigenpair(induced_subgraph(graph, region_of == r))[0] for r in range(num_regions)]
