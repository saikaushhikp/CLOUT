"""R1 helpers (experiments/r1_manifesto_round0.py): blends, scoring, batched PG and the ternary grid."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import r1_manifesto_round0 as r1  # noqa: E402

from dumbocracy.strategies.manifesto_opt import expected_share  # noqa: E402

BETA, Z = 20.0, 0.825


def _toy(n=400, seed=0):
    """Tiny synthetic electorate: 4 regions of unequal size, Dirichlet voters around distinct centres."""
    g = torch.Generator().manual_seed(seed)
    sizes = [160, 110, 80, 50]
    centres = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1],
                            [0.1, 0.1, 0.2, 0.4, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float64)
    region_of = torch.cat([torch.full((s,), r, dtype=torch.long) for r, s in enumerate(sizes)])
    alpha = 40 * centres[region_of]
    v = torch._standard_gamma(alpha, generator=g)
    v = v / v.sum(-1, keepdim=True)
    return v, region_of, sizes


def test_blends_on_simplex_and_names():
    v, reg, sizes = _toy()
    names, b, meta = r1.scenario_manifestos(v, reg, sizes)
    assert len(names) == 19 and b.shape == (19, 5)
    assert torch.allclose(b.sum(-1), torch.ones(19, dtype=b.dtype), atol=1e-9)
    assert bool((b >= 0).all())
    # the 4-region blend is the population mean; R1 = largest region
    means = r1.regional_means(v, reg)
    full = r1.blend(means, torch.tensor(sizes), [0, 1, 2, 3])
    assert torch.allclose(full, v.mean(0), atol=1e-12)
    assert meta["single_R1"]["regions"] == ["N"] and meta["single_R4"]["regions"] == ["W"]
    assert meta["triple_R1R2R3"]["majority"] and not meta["single_R4"]["majority"]
    assert abs(meta["pair_R1R2"]["voter_share"] - 270 / 400) < 1e-12


def test_score_batch_matches_library_and_hist_sums_to_n():
    v, reg, sizes = _toy()
    _, b, _ = r1.scenario_manifestos(v, reg, sizes)
    sc = r1.score_batch(v.unsqueeze(0), reg.unsqueeze(0), b.unsqueeze(0), BETA, Z)
    assert torch.allclose(sc["E"][0], expected_share(v, b, BETA, Z), atol=1e-12)
    assert bool((sc["hist"].sum(-1) == v.shape[0]).all())
    # E is the size-weighted average of the regional supports
    w = torch.tensor(sizes, dtype=v.dtype) / sum(sizes)
    assert torch.allclose((sc["region_P"][0] * w).sum(-1), sc["E"][0], atol=1e-12)
    # S = 1 (manifesto equals a voter) lands in the last bin
    one = r1.score_batch(v[:1].unsqueeze(0), reg[:1].unsqueeze(0), v[:1].unsqueeze(0), BETA, Z)
    assert int(one["hist"][0, 0, -1]) == 1


def test_bstar_beats_every_variant():
    v, reg, sizes = _toy()
    _, b, _ = r1.scenario_manifestos(v, reg, sizes)
    path, trace = r1.pg_ascent(v.unsqueeze(0), b.unsqueeze(0), BETA, Z, iterations=150)
    bstar, E, _, _ = r1.best_iterate(path, trace)
    assert abs(float(bstar.sum()) - 1) < 1e-9 and bool((bstar >= 0).all())
    exact = expected_share(v, b, BETA, Z)
    assert float(E[0]) >= float(exact.max()) - 1e-12
    assert abs(float(expected_share(v, bstar[0], BETA, Z)) - float(E[0])) < 1e-12


def test_ternary_points_and_scenario_threshold():
    pts = r1.ternary_points(10)
    assert pts.shape == (66, 3)
    assert torch.allclose(pts.sum(-1), torch.ones(66, dtype=pts.dtype))
    # archetypes A0 vs A1 (TV 0.35) → divided; A5 vs A5 → 0
    assert abs(r1.tv_two_largest([4, 3, 2, 1], [0, 1, 2, 3]) - 0.35) < 1e-12
    assert r1.tv_two_largest([4, 3, 2, 1], [5, 5, 0, 1]) == 0.0
