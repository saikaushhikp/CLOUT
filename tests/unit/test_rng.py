import pytest
import torch

from dumbocracy.rng import beta, dirichlet, gamma


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("k", [0.3, 1.0, 2.5, 12.0])
def test_gamma_moments(k):
    x = gamma(torch.full((200_000,), k, dtype=torch.float64), gen(1))
    assert torch.all(x > 0)
    assert abs(float(x.mean()) - k) < 0.02 * max(k, 1)
    assert abs(float(x.var()) - k) < 0.05 * max(k, 1)


def test_dirichlet_mean_and_simplex():
    alpha = torch.tensor([4.0, 1.0, 2.0, 2.0, 1.0], dtype=torch.float64).expand(100_000, 5)
    x = dirichlet(alpha, gen(2))
    assert torch.allclose(x.sum(1), torch.ones(100_000, dtype=torch.float64))
    assert torch.allclose(x.mean(0), alpha[0] / alpha[0].sum(), atol=3e-3)


def test_beta_mean():
    x = beta(torch.full((100_000,), 6.0), torch.full((100_000,), 3.0), gen(3))
    assert abs(float(x.mean()) - 6 / 9) < 3e-3


def test_reproducible_and_rejects_bad_input():
    a = gamma(torch.full((50,), 2.0), gen(9))
    assert torch.equal(a, gamma(torch.full((50,), 2.0), gen(9)))
    with pytest.raises(ValueError):
        gamma(torch.tensor([0.0, 1.0]), gen())
