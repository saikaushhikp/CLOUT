"""Free global statistic, paid sampled-survey queries and free polls (spec §6.2–§6.4).

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
    """Exact % of ALL voters in ``region`` with ``v_i[sector_b] > v_i[sector_a]``.

    Since R12 the game answers paid queries with :func:`sample_query`; this exact census value is kept
    for tests, analysis and oracle tools (it is the expectation of a sampled answer).
    """
    if sector_a == sector_b:
        raise ValueError("sector_a and sector_b must differ")
    in_region = region_of == region
    size = int(in_region.sum())
    if size == 0:
        raise ValueError(f"region {region} is empty")
    hits = int(((v[:, sector_b] > v[:, sector_a]) & in_region).sum())
    return 100.0 * hits / size


def query_sample_size(region_size: int, fraction: float) -> int:
    """R12: ``m = max(1, round(f · n_r))`` (half-up rounding), never more than ``n_r``."""
    n = int(region_size)
    if n < 1:
        raise ValueError("region_size must be >= 1")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    return max(1, min(n, int(math.floor(float(fraction) * n + 0.5))))


def sample_query(v: torch.Tensor, region_of: torch.Tensor, sector_a: int, sector_b: int, region: int,
                 fraction: float, generator: torch.Generator) -> tuple[int, int, int]:
    """Spec §6.3 (R12): survey a fresh uniform sample WITHOUT replacement of ``m = max(1, round(f·n_r))``
    voters of ``region`` and count those with ``v_i[sector_b] > v_i[sector_a]``.

    ``generator`` (survey stream, keyed ``("survey", round, player, query_index)``) must live on ``v``'s
    device; the sample is ``randperm(n_r)[:m]`` over the region's voters in index order.
    Returns ``(yes, m, n_r)`` — integers only.
    """
    if sector_a == sector_b:
        raise ValueError("sector_a and sector_b must differ")
    members = torch.nonzero(region_of == region, as_tuple=False).reshape(-1)
    size = int(members.numel())
    if size == 0:
        raise ValueError(f"region {region} is empty")
    m = query_sample_size(size, fraction)
    perm = torch.randperm(size, generator=generator, device=members.device)[:m]
    sample = members[perm].to(v.device)
    yes = int((v[sample, sector_b] > v[sample, sector_a]).sum())
    return yes, m, size


def query_moe(yes: int, m: int, region_size: int | None = None, fpc: bool = True) -> float:
    """95% margin of error in percentage points (spec §6.3, R12).

    ``MoE = 1.96 · 100 · sqrt(p̃(1 − p̃)/m · (n_r − m)/(n_r − 1))`` with ``p̃ = (yes + 1)/(m + 2)`` (so the
    MoE is never 0 for a proper sample). The finite-population factor is applied only when ``fpc`` and
    ``n_r > 1`` (with m = n_r it is 0: the sample is a census). Pooled answers use ``fpc=False``.
    """
    if m < 1 or not 0 <= yes <= m:
        raise ValueError("need m >= 1 and 0 <= yes <= m")
    p = (yes + 1.0) / (m + 2.0)
    var = p * (1.0 - p) / m
    if fpc and region_size is not None and region_size > 1:
        if m > region_size:
            raise ValueError("sample larger than the region")
        var *= (region_size - m) / (region_size - 1.0)
    return POLL_Z * 100.0 * math.sqrt(var)


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
