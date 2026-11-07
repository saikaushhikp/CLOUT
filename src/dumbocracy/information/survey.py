"""Free global statistic, paid region queries and free polls (spec §6.2–§6.4).

All functions are vectorised over voters (no Python loops over N) and read the *current* hidden
state ``v`` [N, 5]; they return only aggregate, public-safe numbers.
"""
from __future__ import annotations

import math

import torch

from ..sectors import NUM_SECTORS, SECTORS
from ..voting import similarity, vote_probability

POLL_Z = 1.96  # 95% normal-approximation margin of error (spec §6.4)


def global_top3_pct(v: torch.Tensor) -> dict[str, float]:
    """Spec §6.2: % of ALL voters whose top-3 sectors (by v_i) include sector k; sums to 300.

    Ties are broken by sector index (a stable descending sort keeps the lower index first), which
    is exactly what ``topk`` intends but does not guarantee.
    """
    if v.ndim != 2 or v.shape[1] != NUM_SECTORS or v.shape[0] == 0:
        raise ValueError(f"v must have shape [N>0, {NUM_SECTORS}], got {list(v.shape)}")
    order = torch.sort(v, dim=1, descending=True, stable=True).indices[:, :3]
    counts = torch.bincount(order.reshape(-1), minlength=NUM_SECTORS).to(torch.float64)
    pct = (100.0 * counts / v.shape[0]).cpu().tolist()
    return {name: float(p) for name, p in zip(SECTORS, pct)}


def query_pct(v: torch.Tensor, region_of: torch.Tensor, sector_a: int, sector_b: int, region: int) -> float:
    """Spec §6.3: exact % of voters in ``region`` with ``v_i[sector_b] > v_i[sector_a]``."""
    if sector_a == sector_b:
        raise ValueError("sector_a and sector_b must differ")
    in_region = region_of == region
    size = int(in_region.sum())
    if size == 0:
        raise ValueError(f"region {region} is empty")
    hits = int(((v[:, sector_b] > v[:, sector_a]) & in_region).sum())
    return 100.0 * hits / size


def poll_moe(estimate: float, sample_size: int) -> float:
    """``MoE = 1.96 · sqrt(p̂(1 − p̂)/n)`` (spec §6.4)."""
    return POLL_Z * math.sqrt(max(estimate * (1.0 - estimate), 0.0) / sample_size)


def run_poll(v: torch.Tensor, manifesto: torch.Tensor, sensitivity: float, threshold: float, n: int,
             generator: torch.Generator) -> tuple[float, float]:
    """Spec §6.4: sample ``min(n, N)`` voters uniformly without replacement; each says "yes" w.p. P_i.

    ``generator`` (poll stream, keyed ``("poll", round, player)``) must live on ``v``'s device.
    Returns ``(p̂, MoE)``.
    """
    num_voters = int(v.shape[0])
    size = min(int(n), num_voters)
    if size <= 0:
        raise ValueError("poll size must be >= 1")
    sample = torch.randperm(num_voters, generator=generator, device=v.device)[:size]
    prob = vote_probability(similarity(v[sample], manifesto.to(device=v.device, dtype=v.dtype)),
                            sensitivity, threshold)
    yes = torch.rand(size, generator=generator, device=v.device, dtype=prob.dtype) < prob
    estimate = float(yes.sum()) / size
    return estimate, poll_moe(estimate, size)
