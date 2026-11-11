"""Helpers of experiments/r2_drift_no_campaign.py (R2: opinion drift with no campaign)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from r2_drift_no_campaign import consensus_limit, drift_metrics, social_run  # noqa: E402
from dumbocracy.electorate import Graph  # noqa: E402


def cycle(n: int) -> Graph:
    src = torch.arange(n)
    return Graph.from_edges(n, src, (src + 1) % n)


def random_simplex(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 5, generator=g, dtype=torch.float64)
    return x / x.sum(-1, keepdim=True)


def test_metrics_zero_for_identical_voters():
    v = torch.full((1, 2, 8, 5), 0.2, dtype=torch.float64)
    region = torch.arange(8).remainder(4).expand(2, 8)
    b = torch.full((2, 1, 5), 0.2, dtype=torch.float64)
    m = drift_metrics(v, v, region, b, torch.ones(2, 8, dtype=torch.bool), 20.0, 0.825)
    for k in ("disagreement", "within", "between", "disagreement_free", "moved", "mean_shift", "region_drift"):
        assert torch.allclose(m[k], torch.zeros_like(m[k])), k
    assert torch.allclose(m["won_0"], torch.ones_like(m["won_0"]))


def test_between_and_within_by_hand():
    # region r: all voters at vertex e_r (r < 4) -> within 0, every pair of region means at TV 1
    v = torch.zeros(1, 1, 8, 5, dtype=torch.float64)
    region = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    v[0, 0, torch.arange(8), region[0]] = 1.0
    b = torch.full((1, 1, 5), 0.2, dtype=torch.float64)
    m = drift_metrics(v, v, region, b, torch.ones(1, 8, dtype=torch.bool), 20.0, 0.825)
    assert float(m["within"]) == 0.0
    assert abs(float(m["between"]) - 1.0) < 1e-12
    assert abs(float(m["disagreement"]) - 0.75) < 1e-12        # TV(e_r, (¼,¼,¼,¼,0)) = ¾


def test_disagreement_decreases_without_zealots():
    n = 8
    g = cycle(n)
    v0 = random_simplex(n)[None]                                  # [1, n, 5]
    weight = torch.full((1, n), 0.3, dtype=torch.float64)         # regular graph + uniform step: doubly stochastic
    frozen = torch.zeros(1, n, dtype=torch.bool)
    region = torch.zeros(1, n, dtype=torch.long)
    b = torch.full((1, 1, 5), 0.2, dtype=torch.float64)
    seen = []
    social_run(v0, g, weight, frozen, 60, on_round=lambda t, v: seen.append(
        float(drift_metrics(v.view(1, 1, n, 5), v0.view(1, 1, n, 5), region, b, ~region.bool(), 20.0, 0.825)["disagreement"])))
    assert all(y <= x + 1e-12 for x, y in zip(seen, seen[1:]))
    assert seen[-1] < 0.05 * seen[0]


def test_zealot_rows_never_change():
    n = 10
    g = cycle(n)
    v0 = random_simplex(n, 1)[None].expand(2, n, 5).clone()
    weight = torch.full((2, n), 0.5, dtype=torch.float64)
    frozen = torch.zeros(2, n, dtype=torch.bool)
    frozen[1, [0, 4]] = True
    weight[1, [0, 4]] = 0.0
    vT = social_run(v0, g, weight, frozen, 100)
    assert torch.equal(vT[1, [0, 4]], v0[1, [0, 4]])
    assert not torch.equal(vT[0, [0, 4]], v0[0, [0, 4]])          # the same voters move when not zealots


def test_consensus_limit_matches_long_run():
    n = 9
    g = cycle(n)
    v0 = random_simplex(n, 2)
    a = torch.linspace(0.1, 0.6, n, dtype=torch.float64)          # heterogeneous steps: π ∝ d/a, not uniform
    vT = social_run(v0[None], g, a[None], torch.zeros(1, n, dtype=torch.bool), 3000)[0]
    lim = consensus_limit(v0, g, a, 1, n)
    assert torch.allclose(vT, lim, atol=1e-9)
    assert not torch.allclose(lim[0], v0.mean(0), atol=1e-3)      # differs from the plain average
