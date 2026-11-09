"""P5: bounded LTL, STL robustness and CTL-style helpers (spec §10.2; lecture "Temporal logics")."""
import math
import random

import pytest
import torch

from dumbocracy.logic.temporal import (
    And, Atom, F, G, Implies, Not, Or, Pred, U, X, exists_eventually, forall_eventually, holds, robustness,
    strategy_eventually,
)

LEAD, WIN, BUDGET = Atom("Lead"), Atom("Victory"), Atom("BudgetLeft")


def trace(*rows):
    return [{"Lead": l, "Victory": v, "BudgetLeft": b} for l, v, b in rows]


T1 = trace((False, False, True), (True, False, True), (True, False, False), (False, True, False))


def test_atoms_and_boolean_connectives():
    assert holds(BUDGET, T1) and not holds(LEAD, T1)
    assert holds(Not(LEAD), T1) and holds(LEAD | BUDGET, T1) and not holds(LEAD & BUDGET, T1)
    assert holds(Implies(LEAD, WIN), T1)                  # vacuous at position 0
    assert not holds(Implies(LEAD, WIN), T1, 1)
    assert holds(~LEAD >> BUDGET, T1)


def test_next_is_strong_on_finite_traces():
    assert holds(X(LEAD), T1, 0) and not holds(X(LEAD), T1, 2)
    assert not holds(X(WIN), T1, 3) and not holds(X(Not(WIN)), T1, 3)   # no successor at the end


def test_bounded_eventually_and_always():
    assert holds(F(WIN), T1) and not holds(F(WIN, 2), T1) and holds(F(WIN, 3), T1)
    assert holds(F(LEAD, 1), T1) and not holds(F(LEAD, 0), T1)
    assert not holds(G(BUDGET), T1) and holds(G(BUDGET, 1), T1) and not holds(G(BUDGET, 2), T1)
    assert holds(G(LEAD, 1), T1, 1) and holds(G(Not(WIN), 2), T1)
    assert holds(G(F(WIN)), T1)                            # every suffix still reaches the final win
    with pytest.raises(ValueError):
        holds(F(WIN, -1), T1)


def test_bounded_until():
    assert holds(U(Not(WIN), WIN), T1)                     # ¬Victory until Victory
    assert not holds(U(Not(WIN), WIN, 2), T1)              # but not within 2 steps
    assert holds(U(BUDGET, LEAD), T1)                      # BudgetLeft holds until Lead at 1
    assert not holds(U(LEAD, WIN), T1)                     # Lead fails at 0 before Victory
    assert holds(U(LEAD, BUDGET), T1)                      # ψ already true at i: no φ needed
    assert not holds(U(BUDGET, Atom("never")), [{"BudgetLeft": True, "never": False}] * 3)


def test_robustness_of_predicates_and_operators():
    tr = [{"margin": m} for m in (-0.10, 0.02, 0.05, -0.01)]
    lead = Pred("margin", 0.0)
    assert robustness(lead, tr) == pytest.approx(-0.10)
    assert robustness(F(lead), tr) == pytest.approx(0.05)
    assert robustness(G(lead), tr) == pytest.approx(-0.10)
    assert robustness(G(lead, 1), tr, 1) == pytest.approx(0.02)
    assert robustness(Pred("margin", 0.0, below=True), tr) == pytest.approx(0.10)
    assert robustness(X(lead), tr, 3) == -math.inf
    assert robustness(U(Pred("margin", -0.2), lead), tr) == pytest.approx(0.05)   # max_j min(ρψ(j), min_{l<j} ρφ(l)) at j = 2
    assert robustness(U(Pred("margin", -0.2), lead, 1), tr) == pytest.approx(0.02)


def _random_formula(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return Pred(rng.choice(["a", "b"]), rng.uniform(-0.5, 0.5), below=rng.random() < 0.3)
    kind = rng.choice(["not", "and", "or", "imp", "X", "F", "G", "U"])
    bound = rng.choice([None, 0, 1, 2, 4])
    sub = lambda: _random_formula(rng, depth - 1)  # noqa: E731
    return {"not": lambda: Not(sub()), "and": lambda: And(sub(), sub()), "or": lambda: Or(sub(), sub()),
            "imp": lambda: Implies(sub(), sub()), "X": lambda: X(sub()), "F": lambda: F(sub(), bound),
            "G": lambda: G(sub(), bound), "U": lambda: U(sub(), sub(), bound)}[kind]()


def test_stl_robustness_sign_agrees_with_boolean_satisfaction():
    rng = random.Random(0)
    checked = 0
    for _ in range(600):
        tr = [{"a": rng.uniform(-1, 1), "b": rng.uniform(-1, 1)} for _ in range(rng.randint(1, 6))]
        phi = _random_formula(rng, 3)
        for i in range(len(tr)):
            rho = robustness(phi, tr, i)
            if rho > 0:
                assert holds(phi, tr, i), (phi, tr, i, rho)
                checked += 1
            elif rho < 0:
                assert not holds(phi, tr, i), (phi, tr, i, rho)
                checked += 1
    assert checked > 1000


def test_ctl_helpers_over_worlds_by_policies():
    win = torch.tensor([[True, False], [False, True]])
    assert exists_eventually(win) and not forall_eventually(win) and not strategy_eventually(win)
    assert strategy_eventually(torch.tensor([[True, False], [True, True]]))
    paths = torch.zeros(2, 2, 3, dtype=torch.bool)
    paths[:, :, 2] = True                                   # every path wins at depth 2
    assert forall_eventually(paths) and strategy_eventually(paths)
    assert not exists_eventually(torch.zeros(3, 4, dtype=torch.bool))
    with pytest.raises(TypeError):
        exists_eventually(torch.zeros(2, 2))
