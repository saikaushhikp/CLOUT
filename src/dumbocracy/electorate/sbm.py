"""Erdős–Rényi stochastic block model with a global sparsity factor (spec §2.2, decisions R4/R5).

The owner's "10 numbers" are ER connection probabilities: ``p_r`` inside region r and ``p_rs``
between regions r and s. One scalar ``s`` multiplies all of them so the expected mean degree is
``k̄``; within each block edges are independent Bernoulli(``s·p``). Sampling is O(E): small blocks
use an exact Bernoulli mask over all pairs, large blocks draw a Binomial edge count and uniform
distinct pair indices (exact up to negligible duplicate collisions when the block is sparse).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..sectors import NUM_REGIONS
from .regions import region_offsets

EXACT_PAIR_LIMIT = 1 << 24  # blocks with at most this many candidate pairs use an exact mask


@dataclass(frozen=True, eq=False)
class Graph:
    """Undirected simple graph: canonical edges (src < dst) plus a symmetric CSR view."""

    num_nodes: int
    src: torch.Tensor      # int64 [E], src < dst, sorted lexicographically
    dst: torch.Tensor      # int64 [E]
    indptr: torch.Tensor   # int64 [N + 1]
    indices: torch.Tensor  # int64 [2E] neighbour lists, sorted within each row
    degree: torch.Tensor   # int64 [N]

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    @property
    def device(self) -> torch.device:
        return self.src.device

    def rows(self) -> torch.Tensor:
        """Row index for every CSR entry (directed edge u -> indices)."""
        return torch.repeat_interleave(torch.arange(self.num_nodes, device=self.device), self.degree)

    @classmethod
    def from_edges(cls, num_nodes: int, src: torch.Tensor, dst: torch.Tensor) -> "Graph":
        """Canonicalise arbitrary undirected pairs: drop loops, orient src < dst, dedupe, build CSR."""
        src = src.to(torch.int64)
        dst = dst.to(torch.int64)
        keep = src != dst
        lo = torch.minimum(src[keep], dst[keep])
        hi = torch.maximum(src[keep], dst[keep])
        key = torch.unique(lo * num_nodes + hi)  # sorted
        lo, hi = key // num_nodes, key % num_nodes
        rows = torch.cat([lo, hi])
        cols = torch.cat([hi, lo])
        order = torch.argsort(rows * num_nodes + cols)
        degree = torch.bincount(rows, minlength=num_nodes)
        indptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=src.device)
        indptr[1:] = torch.cumsum(degree, 0)
        return cls(num_nodes, lo, hi, indptr, cols[order], degree)


def sample_block_probabilities(intra_range: tuple[float, float], inter_range: tuple[float, float],
                               generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the 4 intra probabilities p_r and the symmetric 4x4 inter matrix p_rs (zero diagonal)."""
    device = generator.device
    lo, hi = intra_range
    p_r = lo + (hi - lo) * torch.rand(NUM_REGIONS, generator=generator, dtype=torch.float64, device=device)
    lo, hi = inter_range
    p_rs = torch.zeros(NUM_REGIONS, NUM_REGIONS, dtype=torch.float64, device=device)
    for r in range(NUM_REGIONS):
        for s in range(r + 1, NUM_REGIONS):
            value = lo + (hi - lo) * torch.rand(1, generator=generator, dtype=torch.float64, device=device)
            p_rs[r, s] = p_rs[s, r] = value[0]
    return p_r, p_rs


def block_probability_matrix(sizes: torch.Tensor, p_r: torch.Tensor, p_rs: torch.Tensor,
                             mean_degree: float) -> tuple[torch.Tensor, float, bool]:
    """Return (P [4x4] edge probabilities, sparsity factor s, capped flag).

    s = k̄ N / (Σ_r p_r n_r (n_r − 1) + Σ_{r≠s} p_rs n_r n_s), reduced if any s·p would exceed 1.
    """
    n = sizes.to(torch.float64).cpu()
    raw = p_rs.cpu().clone()
    raw.fill_diagonal_(0.0)
    raw += torch.diag(p_r.cpu())
    pairs = torch.outer(n, n)
    pairs.fill_diagonal_(0.0)
    pairs += torch.diag(n * (n - 1))
    denominator = float((raw * pairs).sum())
    if denominator <= 0 or mean_degree <= 0:
        return torch.zeros_like(raw), 0.0, False
    s = mean_degree * float(n.sum()) / denominator
    capped = False
    if s * float(raw.max()) > 1.0:
        s, capped = 1.0 / float(raw.max()), True
    return raw * s, s, capped


def _triangle_unflatten(flat: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map flat index over pairs i<j (row-major) of an n-node block to (i, j)."""
    two_n_minus_1 = 2 * n - 1
    i = torch.floor((two_n_minus_1 - torch.sqrt((two_n_minus_1 ** 2 - 8 * flat).to(torch.float64))) / 2).to(torch.int64)
    start = lambda row: row * (2 * n - row - 1) // 2  # noqa: E731
    for _ in range(2):  # repair float rounding at row boundaries
        i = torch.where(flat < start(i), i - 1, i)
        i = torch.where(flat >= start(i + 1), i + 1, i)
    j = flat - start(i) + i + 1
    return i, j


def _sample_pair_indices(num_pairs: int, prob: float, generator: torch.Generator) -> torch.Tensor:
    device = generator.device
    if num_pairs <= 0 or prob <= 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    if num_pairs <= EXACT_PAIR_LIMIT:
        mask = torch.rand(num_pairs, generator=generator, device=device) < prob
        return torch.nonzero(mask).squeeze(1)
    count = int(torch.binomial(torch.tensor([float(num_pairs)], dtype=torch.float64, device=device),
                               torch.tensor([prob], dtype=torch.float64, device=device), generator=generator).item())
    flat = torch.randint(0, num_pairs, (count,), generator=generator, device=device, dtype=torch.int64)
    return torch.unique(flat)


def sample_sbm(sizes: torch.Tensor, probabilities: torch.Tensor, generator: torch.Generator) -> Graph:
    """Sample the ER-SBM on the generator's device. Voters are contiguous by region."""
    device = generator.device
    counts = [int(x) for x in sizes.tolist()]
    offsets = region_offsets(sizes)
    num_nodes = sum(counts)
    srcs, dsts = [], []
    for r in range(NUM_REGIONS):
        for s in range(r, NUM_REGIONS):
            prob = float(probabilities[r, s])
            if r == s:
                n = counts[r]
                flat = _sample_pair_indices(n * (n - 1) // 2, prob, generator)
                i, j = _triangle_unflatten(flat, n)
                srcs.append(i + offsets[r])
                dsts.append(j + offsets[r])
            else:
                flat = _sample_pair_indices(counts[r] * counts[s], prob, generator)
                srcs.append(flat // counts[s] + offsets[r])
                dsts.append(flat % counts[s] + offsets[s])
    src = torch.cat(srcs) if srcs else torch.empty(0, dtype=torch.int64, device=device)
    dst = torch.cat(dsts) if dsts else torch.empty(0, dtype=torch.int64, device=device)
    return Graph.from_edges(num_nodes, src, dst)


def block_edge_counts(graph: Graph, region_of: torch.Tensor) -> torch.Tensor:
    """Realised undirected edge counts per region block, symmetric int64 [4x4] (CPU)."""
    rs, rd = region_of[graph.src], region_of[graph.dst]
    lo, hi = torch.minimum(rs, rd), torch.maximum(rs, rd)
    flat = torch.bincount(lo * NUM_REGIONS + hi, minlength=NUM_REGIONS * NUM_REGIONS).reshape(NUM_REGIONS, NUM_REGIONS).cpu()
    return flat + flat.T - torch.diag(torch.diag(flat))
