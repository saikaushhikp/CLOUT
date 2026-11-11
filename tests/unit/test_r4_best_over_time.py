"""Helpers of experiments/r4_best_over_time.py (R4: does the best manifesto stay the best as opinions drift?)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from r4_best_over_time import (  # noqa: E402
    evaluate, first_crossover, first_reach, optimise_rounds, record_states, regret_matrix,
)
from dumbocracy.electorate import Graph  # noqa: E402

BETA, Z = 20.0, 0.825


def toy_voters(n_per: int = 40, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Four regions, each a Dirichlet cloud around a different corner-ish point → v [n, 5], region_of, sizes."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[.5, .2, .1, .1, .1], [.1, .5, .2, .1, .1], [.1, .1, .5, .2, .1], [.2, .1, .1, .1, .5]],
                           dtype=torch.float64)
    v = torch.cat([torch.distributions.Dirichlet(30 * centres[r]).sample((n_per,)) for r in range(4)])
    torch.manual_seed(seed)
    return v, torch.arange(4).repeat_interleave(n_per), [n_per] * 4


def ring(n: int) -> Graph:
    src = torch.arange(n)
    return Graph.from_edges(n, src, (src + 7) % n)


def run_toy(step: float, rounds=(0, 1, 2, 3, 5)):
    v, reg, sizes = toy_voters()
    n = v.shape[0]
    states = record_states(v[None], ring(n), torch.full((1, n), step, dtype=v.dtype), torch.zeros(1, n, dtype=torch.bool),
                           rounds)
    states = {t: s.view(1, n, 5) for t, s in states.items()}
    opt = optimise_rounds(states, reg[None], [sizes], BETA, Z, rounds, iterations=150)
    E = evaluate(states, opt["b"].permute(1, 0, 2), BETA, Z, rounds)[:, :, 0]          # [S, T]
    return opt, E, list(range(len(rounds)))


def test_regret_diagonal_zero_and_nonnegative_with_drift():
    opt, E, didx = run_toy(0.3)
    R = regret_matrix(E, didx)
    assert torch.allclose(torch.diagonal(R), torch.zeros(len(didx), dtype=R.dtype))
    assert float(R.min()) >= -2e-3                          # optimiser tolerance
    # the evaluated diagonal equals the optimiser's own best exact E
    assert torch.allclose(torch.diagonal(E), opt["E"][:, 0], atol=1e-9)


def test_zero_drift_keeps_the_same_optimum():
    opt, E, didx = run_toy(0.0)
    b = opt["b"][:, 0]
    assert float(0.5 * (b - b[0]).abs().sum(-1).max()) < 1e-2
    assert float((opt["E"][:, 0] - opt["E"][0, 0]).abs().max()) < 1e-4
    assert float(regret_matrix(E, didx).abs().max()) < 1e-4


def test_record_states_round0_is_input_and_zero_step_is_static():
    v, _, _ = toy_voters()
    n = v.shape[0]
    st = record_states(v[None], ring(n), torch.zeros(1, n, dtype=v.dtype), torch.zeros(1, n, dtype=torch.bool), (0, 4))
    assert torch.equal(st[0][0], v) and torch.allclose(st[4][0], v)


def test_regret_matrix_by_hand_and_nan_off_design():
    E = torch.tensor([[0.50, 0.40, 0.30], [0.45, 0.48, 0.41]])     # designs at round indices 0 and 1 of 3 rounds
    R = regret_matrix(E, [0, 1])
    assert torch.allclose(R[:, :2], torch.tensor([[0.0, 0.08], [0.05, 0.0]]))
    assert torch.isnan(R[:, 2]).all()


def test_first_crossover_and_first_reach():
    rounds = [0, 1, 2, 3, 5, 10]
    a = [0.40, 0.39, 0.38, 0.37, 0.35, 0.30]
    b = [0.38, 0.385, 0.38, 0.375, 0.37, 0.36]
    assert first_crossover(a, b, rounds) == 3                    # tie at round 2 is not a crossover
    assert first_crossover(b, a, rounds) == 0
    assert first_crossover(a, [x - 1 for x in a], rounds) is None
    assert first_reach([0.0, 0.01, 0.06, 0.04], [0, 1, 2, 3], 0.05) == 2
    assert first_reach([0.0, 0.01], [0, 1], 0.05) is None
