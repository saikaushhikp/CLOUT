import torch

from dumbocracy.strategies.manifesto_opt import (
    expected_share, optimise_manifesto, project_simplex,
)


def test_project_simplex_known_cases():
    y = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [2.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5],
                      [-1.0, 0.3, 0.9, 0.1, 0.0]], dtype=torch.float64)
    x = project_simplex(y)
    assert torch.allclose(x[0], y[0])
    assert torch.allclose(x[1], torch.tensor([1.0, 0, 0, 0, 0], dtype=torch.float64))
    assert torch.allclose(x[2], torch.full((5,), 0.2, dtype=torch.float64))
    assert torch.allclose(x.sum(-1), torch.ones(4, dtype=torch.float64)) and torch.all(x >= 0)


def test_project_simplex_is_the_nearest_point():
    gen = torch.Generator().manual_seed(0)
    y = torch.randn(200, 5, generator=gen, dtype=torch.float64)
    x = project_simplex(y)
    # Optimality (KKT): (y − x)·(z − x) ≤ 0 for every z in the simplex; test the vertices.
    vertices = torch.eye(5, dtype=torch.float64)
    inner = ((y - x).unsqueeze(1) * (vertices.unsqueeze(0) - x.unsqueeze(1))).sum(-1)
    assert float(inner.max()) < 1e-10
    batched = project_simplex(y.reshape(20, 10, 5)).reshape(200, 5)
    assert torch.allclose(batched, x)


def test_expected_share_exact_and_batched():
    v = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]], dtype=torch.float64)
    b = torch.tensor([1.0, 0, 0, 0, 0], dtype=torch.float64)
    share = expected_share(v, b, 12.0, 0.75)
    expected = (torch.sigmoid(torch.tensor(12 * 0.25)) + torch.sigmoid(torch.tensor(12 * -0.75))) / 2
    assert torch.isclose(share, expected.double())
    both = expected_share(v, torch.stack([b, b]), 12.0, 0.75)
    assert both.shape == (2,) and torch.allclose(both, share.expand(2))


def test_optimiser_finds_the_cluster_and_satisfies_kkt():
    gen = torch.Generator().manual_seed(1)
    centre = torch.tensor([0.1, 0.5, 0.2, 0.1, 0.1], dtype=torch.float64)
    alpha = 60 * centre
    g = torch._standard_gamma(alpha.expand(4000, 5).contiguous())
    v = g / g.sum(-1, keepdim=True)
    sol = optimise_manifesto(v, 12.0, 0.75)
    assert sol.kkt_residual < 1e-3
    assert float((sol.manifesto - centre).abs().sum()) < 0.15
    assert sol.share > float(expected_share(v, torch.full((5,), 0.2, dtype=torch.float64), 12.0, 0.75))


def _two_cluster_electorate(n=3000, seed=4):
    gen = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.4, 0.1, 0.1, 0.25, 0.15], [0.15, 0.2, 0.4, 0.1, 0.15],
                            [0.25, 0.1, 0.1, 0.15, 0.4], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float64)
    region_of = torch.arange(4).repeat_interleave(n // 4)
    alpha = 60 * centres[region_of]
    from dumbocracy.rng import dirichlet
    return dirichlet(alpha, gen), region_of


def test_admm_matches_centralised_projected_gradient():
    from dumbocracy.strategies.manifesto_opt import admm_manifesto
    v, region_of = _two_cluster_electorate()
    central = optimise_manifesto(v, 20.0, 0.825)
    admm = admm_manifesto(v, region_of, 20.0, 0.825, start=central.manifesto)
    assert abs(admm.share - central.share) < 2e-3
    assert float((admm.manifesto - central.manifesto).abs().sum()) < 0.05
    assert admm.primal_residuals[-1] < 1e-3          # local copies agree with the consensus
    assert torch.allclose(admm.manifesto.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_admm_from_uniform_start_improves_and_converges():
    from dumbocracy.strategies.manifesto_opt import admm_manifesto
    v, region_of = _two_cluster_electorate(seed=5)
    uniform = torch.full((5,), 0.2, dtype=torch.float64)
    admm = admm_manifesto(v, region_of, 20.0, 0.825)
    assert admm.share >= float(expected_share(v, uniform, 20.0, 0.825)) - 1e-9
    assert admm.primal_residuals[-1] < admm.primal_residuals[0] + 1e-12 or admm.primal_residuals[-1] < 1e-3
    assert admm.dual_residuals[-1] < 1e-2
