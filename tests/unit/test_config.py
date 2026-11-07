from decimal import Decimal

import pytest
import torch

from dumbocracy.config import (
    SEED_STREAMS, ConfigError, ElectorateConfig, GameConfig, SeedStreams, WorkerConfig, resolve_device,
)


def test_defaults_validate_and_record_is_json_safe():
    import json
    cfg = GameConfig()
    assert cfg.rounds == 5 and cfg.budget == Decimal("500.00")
    record = cfg.to_record()
    json.dumps(record)
    assert record["budget"] == "500.00"
    assert record["electorate"]["intra_prob_range"] == [0.5, 0.9]
    assert record["electorate"]["inter_prob_range"] == [0.01, 0.2]


@pytest.mark.parametrize("kwargs", [
    {"rounds": 0}, {"rounds": 11}, {"budget": Decimal("-1")}, {"budget": 100},
    {"dtype": "float16"}, {"device": "tpu"}, {"seed": True},
])
def test_invalid_game_config(kwargs):
    with pytest.raises(ConfigError):
        GameConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"num_voters": 3}, {"intra_prob_range": (0.9, 0.5)}, {"inter_prob_range": (0.0, 0.2)}, {"mean_degree": -1},
    {"zealot_fraction": 1.5}, {"dirichlet_concentration": 0}, {"num_voters": 7}, {"min_region_fraction": 0.3},
])
def test_invalid_electorate_config(kwargs):
    with pytest.raises(ConfigError):
        ElectorateConfig(**kwargs)


def test_contacts_per_worker():
    assert WorkerConfig().contacts_per_worker(10_000) == 25
    assert WorkerConfig().contacts_per_worker(10) == 1


def test_seed_streams_are_stable_independent_and_keyed():
    s = SeedStreams(7)
    assert s.derive("graph") == SeedStreams(7).derive("graph")
    assert s.derive("graph") != s.derive("preferences")
    assert s.derive("workers", 1, "p0") != s.derive("workers", 2, "p0")
    assert len({s.derive(name) for name in SEED_STREAMS}) == len(SEED_STREAMS)
    a = torch.rand(4, generator=s.generator("poll", 3))
    b = torch.rand(4, generator=s.generator("poll", 3))
    assert torch.equal(a, b)
    with pytest.raises(ConfigError):
        s.derive("nope")


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError):
            resolve_device("cuda:0")
