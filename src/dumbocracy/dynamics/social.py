"""Social aggregation ``Agg_i(v)`` for the preference update (spec §4).

Both rules read the synchronous snapshot v(t) and work on the symmetric CSR view of the
:class:`~dumbocracy.electorate.sbm.Graph` (w_ij = 1 on edges). Inputs have shape ``[..., N, 5]``:
optional leading batch dimensions share the one graph. Memory is O(E · batch); no N × N tensor is
ever formed. Everything runs on the device of ``v`` (the graph must live on the same device).
"""
from __future__ import annotations

import torch

from ..electorate.sbm import Graph

SOCIAL_RULES = ("degroot", "wmsr")


def _check_input(v: torch.Tensor, graph: Graph) -> None:
    if v.ndim < 2 or v.shape[-2] != graph.num_nodes:
        raise ValueError(f"v must have shape [..., N={graph.num_nodes}, K], got {tuple(v.shape)}")


def aggregate_degroot(v: torch.Tensor, graph: Graph) -> torch.Tensor:
    """DeGroot neighbour mean ``Σ_j w_ij v_j / Σ_j w_ij`` (spec §4); isolated voters keep own v_i."""
    _check_input(v, graph)
    rows = graph.rows()
    neighbour_values = v.index_select(-2, graph.indices)          # [..., 2E, K]
    total = torch.zeros_like(v).index_add_(-2, rows, neighbour_values)
    degree = graph.degree.to(v.dtype).unsqueeze(-1)               # [N, 1]
    mean = total / degree.clamp_min(1)
    return torch.where((graph.degree > 0).unsqueeze(-1), mean, v)


def aggregate_wmsr(v: torch.Tensor, graph: Graph, trim: int) -> torch.Tensor:
    """W-MSR resilient aggregation adapted to the simplex (spec §4, LeBlanc et al. 2013).

    For voter i, neighbours are ranked by TV distance ``d_ij = ½‖v_j − v_i‖₁``; the ``trim`` (F)
    farthest are discarded and the rest averaged. If ``deg_i ≤ F`` the voter keeps its own v_i.
    Ties in distance are broken by ascending neighbour index (CSR order), so the result is
    deterministic. Implementation: a two-pass stable segment sort (by distance, then by row) gives
    each CSR entry its rank inside its row; entries with ``rank < deg_i − F`` are kept.
    """
    if not isinstance(trim, int) or isinstance(trim, bool) or trim < 0:
        raise ValueError(f"trim must be an integer >= 0, got {trim!r}")
    _check_input(v, graph)
    if trim == 0:
        return aggregate_degroot(v, graph)

    num_nodes, num_sectors = v.shape[-2], v.shape[-1]
    batch_shape = v.shape[:-2]
    flat = v.reshape(-1, num_nodes, num_sectors)                 # [B, N, K]
    batch = flat.shape[0]
    rows, cols = graph.rows(), graph.indices                     # [2E] each, rows non-decreasing
    num_entries = rows.numel()

    distance = 0.5 * (flat.index_select(1, cols) - flat.index_select(1, rows)).abs().sum(-1)  # [B, 2E]
    by_distance = torch.argsort(distance, dim=-1, stable=True)                                 # [B, 2E]
    by_row = torch.argsort(rows[by_distance], dim=-1, stable=True)
    order = by_distance.gather(-1, by_row)  # sorted position k -> CSR entry; row of position k is rows[k]

    rank = torch.arange(num_entries, device=v.device) - graph.indptr[rows]                    # [2E]
    kept_per_row = (graph.degree - trim).clamp_min(0)                                          # [N]
    keep = rank < kept_per_row[rows]                                                           # [2E]

    neighbours = cols[order]                                                                   # [B, 2E]
    values = flat.gather(1, neighbours.unsqueeze(-1).expand(batch, num_entries, num_sectors))
    values = values * keep.unsqueeze(-1).to(v.dtype)
    target = (torch.arange(batch, device=v.device).unsqueeze(-1) * num_nodes + rows).reshape(-1)
    total = torch.zeros(batch * num_nodes, num_sectors, dtype=v.dtype, device=v.device)
    total.index_add_(0, target, values.reshape(-1, num_sectors))
    total = total.reshape(batch, num_nodes, num_sectors)

    mean = total / kept_per_row.to(v.dtype).clamp_min(1).unsqueeze(-1)
    result = torch.where((kept_per_row > 0).unsqueeze(-1), mean, flat)
    return result.reshape(*batch_shape, num_nodes, num_sectors)


def aggregate(v: torch.Tensor, graph: Graph, rule: str, trim: int) -> torch.Tensor:
    """Dispatch on ``DynamicsConfig.social_rule`` ("degroot" | "wmsr"; ``trim`` = F for W-MSR)."""
    if rule == "degroot":
        return aggregate_degroot(v, graph)
    if rule == "wmsr":
        return aggregate_wmsr(v, graph, trim)
    raise ValueError(f"unknown social rule {rule!r}; expected one of {SOCIAL_RULES}")
