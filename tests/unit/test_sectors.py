import math

import pytest
import torch

from dumbocracy.sectors import (
    ARCHETYPES, NUM_REGIONS, NUM_SECTORS, PERSONALITIES, archetype_means,
    personality_beta_params, region_index, sector_index,
)


def test_archetype_means_are_on_the_simplex_and_distinct():
    means = archetype_means(dtype=torch.float64)
    assert means.shape == (6, NUM_SECTORS)
    assert torch.all(means > 0)
    assert torch.allclose(means.sum(dim=1), torch.ones(6, dtype=torch.float64))
    assert len({a.mean for a in ARCHETYPES}) == len(ARCHETYPES)
    assert len(ARCHETYPES) >= NUM_REGIONS  # drawn without replacement (U11)


def test_personalities_have_positive_beta_parameters():
    params = personality_beta_params()
    assert params.shape == (len(PERSONALITIES), 3, 2)
    assert torch.all(params > 0)
    assert len(PERSONALITIES) == NUM_REGIONS  # one per region via a permutation


def test_lookup_helpers():
    assert sector_index("Agri") == 2 and sector_index("Education") == 0
    assert region_index("W") == 3 and region_index("North") == 0
    with pytest.raises(KeyError):
        sector_index("Defence")
