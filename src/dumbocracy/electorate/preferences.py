"""Archetype assignment and Dirichlet preference draws (spec §2.3)."""
from __future__ import annotations

import torch

from ..rng import dirichlet
from ..sectors import ARCHETYPES, NUM_REGIONS, archetype_means


def sample_assignment(generator: torch.Generator) -> torch.Tensor:
    """Injective archetype index per region: int64 [4] drawn without replacement from 6 (U11)."""
    return torch.randperm(len(ARCHETYPES), generator=generator, device=generator.device)[:NUM_REGIONS].cpu()


def sample_preferences(region_of: torch.Tensor, assignment: torch.Tensor, concentration: float,
                       generator: torch.Generator, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """v_i(0) ~ Dirichlet(κ · μ_σ(region_i)); returns [N, 5] on the generator's device."""
    device = generator.device
    means = archetype_means(device=device, dtype=torch.float64)[assignment.to(device)]  # [4, 5]
    alpha = concentration * means[region_of.to(device)]
    return dirichlet(alpha, generator).to(dtype)
