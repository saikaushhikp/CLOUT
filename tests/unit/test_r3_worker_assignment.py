"""Helpers of experiments/r3_worker_assignment.py: region roles, integer splits, the round-1 worker policy."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import r3_worker_assignment as r3  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import top3_manifesto  # noqa: E402


def test_roles_basic():
    roles = r3.assign_roles([0.30, 0.90, 0.05, 0.55])
    assert roles == {"aligned": 1, "swing": 3, "antagonistic": 2, "other": 0}


@pytest.mark.parametrize("support", [[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8], [0.4, 0.6, 0.6, 0.4],
                                     [0.1, 0.1, 0.1, 0.9]])
def test_roles_distinct_with_ties(support):
    roles = r3.assign_roles(support)
    assert sorted(roles.values()) == [0, 1, 2, 3]
    assert support[roles["aligned"]] == max(support)
    assert support[roles["antagonistic"]] == min(support)
    rest = [r for r in range(4) if r not in (roles["aligned"], roles["antagonistic"])]
    assert abs(support[roles["swing"]] - 0.5) == min(abs(support[r] - 0.5) for r in rest)


def test_roles_all_equal_deterministic():
    assert r3.assign_roles([0.5] * 4) == {"aligned": 0, "swing": 1, "antagonistic": 3, "other": 2}


@pytest.mark.parametrize("K", [0, 1, 3, 4, 7, 10, 25])
@pytest.mark.parametrize("weights", [[1, 1, 1, 1], [3000, 2500, 2400, 2100], [1, 0, 0, 0], [1, 1, 0, 0],
                                     [0.37, 0.21, 0.22, 0.2]])
def test_largest_remainder_sums(K, weights):
    out = r3.largest_remainder(K, weights)
    assert sum(out) == K and all(x >= 0 for x in out)
    assert all(out[i] == 0 for i, w in enumerate(weights) if w == 0)


def test_largest_remainder_tie_priority():
    assert r3.largest_remainder(10, [1, 1, 1, 1], [2, 3, 0, 1]) == (2, 2, 3, 3)
    assert r3.largest_remainder(10, [1, 1, 1, 1]) == (3, 3, 2, 2)


@pytest.mark.parametrize("target", r3.TARGETS)
@pytest.mark.parametrize("K", r3.KS)
def test_target_hires(target, K):
    roles = {"aligned": 2, "swing": 0, "antagonistic": 3, "other": 1}
    sizes = [3000, 2000, 2800, 2200]
    h = r3.target_hires(target, K, roles, sizes)
    assert sum(h) == K
    expect = {"aligned": {2}, "swing": {0}, "antagonistic": {3}, "aligned_swing": {0, 2},
              "even": {0, 1, 2, 3}, "proportional": {0, 1, 2, 3}}[target]
    assert {r for r in range(4) if h[r] > 0} == expect
    if target == "aligned_swing":
        assert h[2] == h[0] == K // 2


@pytest.mark.parametrize("target", ["swing", "aligned_swing", "even"])
def test_policy_round_one_only(target):
    K = 4
    cfg = GameConfig(seed=7, device="cpu", electorate=ElectorateConfig(num_voters=1500))
    e = generate_electorate(cfg, device="cpu")
    b = top3_manifesto(e.preferences)
    sizes = [int(x) for x in e.sizes.tolist()]
    roles = r3.assign_roles(r3.regional_mean_p(e.preferences, b, e.region_of, cfg))
    hires = r3.target_hires(target, K, roles, sizes)
    o = run_batch([e], cfg, b.unsqueeze(0), r3.RoundOneHires(hires), seeds=[7])[0]
    assert tuple(o.allocations[0].hires_by_region) == hires
    assert all(a.total_hires == 0 for a in o.allocations[1:])
    assert all(not any(a.media_regions) for a in o.allocations)
    assert o.spent == Decimal(20) * K == cfg.workers.worker_cost * K
    team = [int(x.numel()) for x in o.workers.worker_ids]
    assert team == list(hires)
    # workers sit in the requested regions only
    for r, ids in enumerate(o.workers.worker_ids):
        assert bool((e.region_of[ids] == r).all())
