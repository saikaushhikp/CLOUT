"""Region personalities, per-voter susceptibilities and zealots (spec §2.4)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..rng import beta
from ..sectors import NUM_REGIONS, personality_beta_params


@dataclass(frozen=True, eq=False)
class Susceptibilities:
    social: torch.Tensor    # α_i  [N]
    media: torch.Tensor     # β_i  [N]
    campaign: torch.Tensor  # γ_i  [N]
    zealot: torch.Tensor    # bool [N]


def sample_personalities(generator: torch.Generator) -> torch.Tensor:
    """Permutation τ: region -> personality index, int64 [4] (CPU)."""
    return torch.randperm(NUM_REGIONS, generator=generator, device=generator.device).cpu()


def sample_susceptibilities(region_of: torch.Tensor, personality: torch.Tensor, zealot_fraction: float,
                            beta_generator: torch.Generator, zealot_generator: torch.Generator,
                            dtype: torch.dtype = torch.float32) -> Susceptibilities:
    device = beta_generator.device
    params = personality_beta_params(device=device, dtype=torch.float64)[personality.to(device)]  # [4, 3, 2]
    per_voter = params[region_of.to(device)]                                                       # [N, 3, 2]
    draws = beta(per_voter[..., 0], per_voter[..., 1], beta_generator)                             # [N, 3]
    zealot = torch.rand(region_of.numel(), generator=zealot_generator, device=zealot_generator.device) < zealot_fraction
    zealot = zealot.to(device)
    draws = torch.where(zealot.unsqueeze(1), torch.zeros_like(draws), draws).to(dtype)
    return Susceptibilities(draws[:, 0].contiguous(), draws[:, 1].contiguous(), draws[:, 2].contiguous(), zealot)
