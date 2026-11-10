"""Synchronous preference update, once per round after all allocations (spec §4).

::

    a_i   = ω · α_i                     (passed in as ``social_weight``; zealots: 0)
    c_i   = Σ_p c_i^p
    scale = 1 if a_i + c_i ≤ 1 else (1 − a_i)/c_i       (applied to every c_i^p)
    v_i  ← (1 − a_i − Σ_p c_i^p) v_i + a_i · Agg_i(v) + Σ_p c_i^p · b_p

The result is a convex combination of simplex points; it is clamped at 0 and renormalised to absorb
float error. Supports leading batch dimensions ``[..., N, 5]``.

P8 (R23): ``pinned[p]`` marks player p's party workers; after the step they are set exactly to ``b_p``
(friends and media never move them, and their v = b feeds the next round's aggregates). Zealots
(``frozen``) keep their own v.
"""
from __future__ import annotations

import torch

SIMPLEX_TOLERANCE = 1e-5


def _broadcast_voter_tensor(x: torch.Tensor, voter_shape: torch.Size, name: str) -> torch.Tensor:
    """Accept ``[N]`` (shared over the batch) or the full ``[..., N]`` voter shape."""
    if x.shape != voter_shape and x.shape != voter_shape[-1:]:
        raise ValueError(f"{name} must have shape [N] or {list(voter_shape)}, got {list(x.shape)}")
    return x.expand(voter_shape)


def step_preferences(v: torch.Tensor, aggregated: torch.Tensor, social_weight: torch.Tensor,
                     campaign_intensity: torch.Tensor, manifestos: torch.Tensor,
                     frozen: torch.Tensor | None = None, pinned: torch.Tensor | None = None) -> torch.Tensor:
    """One synchronous §4 update.

    Args:
        v: current preferences v(t), ``[..., N, K]`` on the simplex.
        aggregated: ``Agg(v(t))`` from :mod:`.social`, same shape as ``v``.
        social_weight: ``a_i = ω·α_i`` in [0, 1], ``[N]`` or ``[..., N]``.
        campaign_intensity: ``c_i^p`` in [0, 1], ``[P, ..., N]`` (``[P, N]`` is broadcast over the
            batch); P may be 0 for a pure social step.
        manifestos: ``b_p`` on the simplex, ``[P, K]``.
        frozen: zealot mask ``[N]`` or ``[..., N]``; frozen voters keep ``v`` bit-for-bit.
        pinned: party-worker masks ``[P, N]`` or ``[P, ..., N]``; pinned voters of player p are set to
            ``b_p`` exactly (applied in player order, before ``frozen``).

    Returns:
        v(t+1), same shape and dtype as ``v``.
    """
    if v.ndim < 2:
        raise ValueError(f"v must have shape [..., N, K], got {list(v.shape)}")
    if aggregated.shape != v.shape:
        raise ValueError(f"aggregated shape {list(aggregated.shape)} != v shape {list(v.shape)}")
    voter_shape = v.shape[:-1]
    num_voters, num_sectors = v.shape[-2], v.shape[-1]

    if manifestos.ndim != 2 or manifestos.shape[1] != num_sectors:
        raise ValueError(f"manifestos must have shape [P, {num_sectors}], got {list(manifestos.shape)}")
    num_players = manifestos.shape[0]
    if num_players and (bool((manifestos < -SIMPLEX_TOLERANCE).any())
                        or bool(((manifestos.sum(-1) - 1).abs() > SIMPLEX_TOLERANCE).any())):
        raise ValueError("manifestos must lie on the probability simplex")

    if campaign_intensity.ndim == 2 and len(voter_shape) > 1:
        first, last = campaign_intensity.shape
        campaign_intensity = campaign_intensity.reshape(first, *([1] * (len(voter_shape) - 1)), last)
    full_shape = torch.Size((num_players, *voter_shape))
    if campaign_intensity.ndim != len(full_shape) or campaign_intensity.shape[0] != num_players \
            or campaign_intensity.shape[-1] != num_voters:
        raise ValueError(f"campaign_intensity must have shape {list(full_shape)} (or [P, N]), "
                         f"got {list(campaign_intensity.shape)}")
    campaign_intensity = campaign_intensity.expand(full_shape)
    if campaign_intensity.numel() and bool(((campaign_intensity < 0) | (campaign_intensity > 1)).any()):
        raise ValueError("campaign_intensity must lie in [0, 1]")

    a = _broadcast_voter_tensor(social_weight, voter_shape, "social_weight").to(v.dtype)
    if bool(((a < 0) | (a > 1)).any()):
        raise ValueError("social_weight must lie in [0, 1]")

    c = campaign_intensity.to(v.dtype)
    total_c = c.sum(0)                                                           # [..., N]
    over = a + total_c > 1
    scale = torch.where(over, (1 - a) / torch.where(over, total_c, torch.ones_like(total_c)),
                        torch.ones_like(total_c))
    c = c * scale
    total_c = c.sum(0)

    pull = torch.einsum("p...n,pk->...nk", c, manifestos.to(v.dtype))           # Σ_p c^p b_p
    new = (1 - a - total_c).unsqueeze(-1) * v + a.unsqueeze(-1) * aggregated + pull
    new = new.clamp_min(0)
    new = new / new.sum(-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)

    if pinned is not None:
        if pinned.ndim == 2 and len(voter_shape) > 1:
            pinned = pinned.reshape(pinned.shape[0], *([1] * (len(voter_shape) - 1)), pinned.shape[-1])
        if pinned.ndim != len(full_shape) or pinned.shape[0] != num_players or pinned.shape[-1] != num_voters:
            raise ValueError(f"pinned must have shape {list(full_shape)} (or [P, N]), got {list(pinned.shape)}")
        pinned = pinned.to(torch.bool).expand(full_shape)
        for p in range(num_players):
            new = torch.where(pinned[p].unsqueeze(-1), manifestos[p].to(v.dtype), new)

    if frozen is not None:
        mask = _broadcast_voter_tensor(frozen.to(torch.bool), voter_shape, "frozen")
        new = torch.where(mask.unsqueeze(-1), v, new)
    return new
