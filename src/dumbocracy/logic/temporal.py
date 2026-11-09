"""Bounded temporal logic over finite per-round traces (spec §10.2; P5; lecture "Temporal logics").

**Traces.** A trace is a finite sequence of states s_0 … s_{n−1} (one per round). A state is a mapping
from atom names to Booleans and from signal names to reals. :func:`game_trace` builds the player's trace
from an observation and an epistemic timeline: state i = round i+1 with atoms ``Lead`` (poll estimate
> 0.5), ``BudgetLeft`` (treasury > 0 at the decision point), ``Viable`` (= possible_victory at t),
``Guaranteed`` (= guaranteed_victory at t), ``Victory`` (official result, final state only — false before),
and the real signal ``margin`` = poll estimate − 0.5 (the final state uses the official vote share).

**LTL on finite traces** (bounded, strong next): with i the current position,

* ``X φ``          — i+1 < n and φ at i+1 (false at the last state);
* ``F[≤k] φ``      — ∃j ∈ [i, min(i+k, n−1)]: φ at j        (k = None: unbounded = to the end);
* ``G[≤k] φ``      — ∀j ∈ [i, min(i+k, n−1)]: φ at j;
* ``φ U[≤k] ψ``    — ∃j ∈ [i, min(i+k, n−1)]: ψ at j and ∀l ∈ [i, j): φ at l;
* ``Not / And / Or / Implies`` as usual.

**STL robustness** (min/max quantitative semantics) on predicates ``Pred(signal, c)`` meaning
``signal > c`` with ρ = signal − c (``below=True``: ``signal < c``, ρ = c − signal): ¬ ↦ −ρ, ∧ ↦ min,
∨ ↦ max, → ↦ max(−ρ₁, ρ₂), F ↦ max over the window, G ↦ min, U ↦ max_j min(ρ_ψ(j), min_{l<j} ρ_φ(l)),
X ↦ ρ at i+1 (−∞ at the end). Boolean atoms have ρ = ±∞. Soundness: ρ > 0 ⇒ satisfied, ρ < 0 ⇒ violated
(tested on random traces).

**CTL-style branching** over the worlds × policies outcome tree (depth = remaining rounds): the player
picks a plan π, nature a world w; a path is (w, π). :func:`exists_eventually` (∃◇Victory) and
:func:`forall_eventually` (∀◇Victory) quantify over all paths; :func:`strategy_eventually` gives the
strategic ∃π ∀w ◇Victory (= guaranteed_victory when Victory is only checked on election day).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import torch

__all__ = ["And", "Atom", "F", "Formula", "G", "Implies", "Not", "Or", "Pred", "U", "X", "exists_eventually",
           "forall_eventually", "game_trace", "holds", "robustness", "strategy_eventually"]

State = Mapping[str, Any]


class Formula:
    """Base class; build with the constructors below. ``&``, ``|``, ``~`` and ``>>`` (implies) are overloaded."""

    def __and__(self, other: "Formula") -> "Formula":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return Or(self, other)

    def __invert__(self) -> "Formula":
        return Not(self)

    def __rshift__(self, other: "Formula") -> "Formula":
        return Implies(self, other)


@dataclass(frozen=True)
class Atom(Formula):
    name: str


@dataclass(frozen=True)
class Pred(Formula):
    signal: str
    threshold: float = 0.0
    below: bool = False


@dataclass(frozen=True)
class Not(Formula):
    phi: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class X(Formula):
    phi: Formula


@dataclass(frozen=True)
class F(Formula):
    phi: Formula
    bound: int | None = None


@dataclass(frozen=True)
class G(Formula):
    phi: Formula
    bound: int | None = None


@dataclass(frozen=True)
class U(Formula):
    left: Formula
    right: Formula
    bound: int | None = None


def _window(i: int, n: int, bound: int | None) -> range:
    if bound is not None and bound < 0:
        raise ValueError("temporal bound must be >= 0")
    end = n - 1 if bound is None else min(n - 1, i + bound)
    return range(i, end + 1)


def holds(phi: Formula, trace: Sequence[State], i: int = 0) -> bool:
    """Boolean satisfaction of ``phi`` at position ``i`` of the finite ``trace`` (module docstring)."""
    n = len(trace)
    if not 0 <= i < n:
        raise IndexError("position outside the trace")
    if isinstance(phi, Atom):
        return bool(trace[i][phi.name])
    if isinstance(phi, Pred):
        x = float(trace[i][phi.signal])
        return x < phi.threshold if phi.below else x > phi.threshold
    if isinstance(phi, Not):
        return not holds(phi.phi, trace, i)
    if isinstance(phi, And):
        return holds(phi.left, trace, i) and holds(phi.right, trace, i)
    if isinstance(phi, Or):
        return holds(phi.left, trace, i) or holds(phi.right, trace, i)
    if isinstance(phi, Implies):
        return (not holds(phi.left, trace, i)) or holds(phi.right, trace, i)
    if isinstance(phi, X):
        return i + 1 < n and holds(phi.phi, trace, i + 1)
    if isinstance(phi, F):
        return any(holds(phi.phi, trace, j) for j in _window(i, n, phi.bound))
    if isinstance(phi, G):
        return all(holds(phi.phi, trace, j) for j in _window(i, n, phi.bound))
    if isinstance(phi, U):
        for j in _window(i, n, phi.bound):
            if holds(phi.right, trace, j):
                return True
            if not holds(phi.left, trace, j):
                return False
        return False
    raise TypeError(f"unknown formula {phi!r}")


def robustness(phi: Formula, trace: Sequence[State], i: int = 0) -> float:
    """STL robustness ρ(φ, trace, i) with min/max semantics (module docstring)."""
    n = len(trace)
    if not 0 <= i < n:
        raise IndexError("position outside the trace")
    if isinstance(phi, Atom):
        return math.inf if trace[i][phi.name] else -math.inf
    if isinstance(phi, Pred):
        x = float(trace[i][phi.signal])
        return phi.threshold - x if phi.below else x - phi.threshold
    if isinstance(phi, Not):
        return -robustness(phi.phi, trace, i)
    if isinstance(phi, And):
        return min(robustness(phi.left, trace, i), robustness(phi.right, trace, i))
    if isinstance(phi, Or):
        return max(robustness(phi.left, trace, i), robustness(phi.right, trace, i))
    if isinstance(phi, Implies):
        return max(-robustness(phi.left, trace, i), robustness(phi.right, trace, i))
    if isinstance(phi, X):
        return robustness(phi.phi, trace, i + 1) if i + 1 < n else -math.inf
    if isinstance(phi, F):
        return max(robustness(phi.phi, trace, j) for j in _window(i, n, phi.bound))
    if isinstance(phi, G):
        return min(robustness(phi.phi, trace, j) for j in _window(i, n, phi.bound))
    if isinstance(phi, U):
        best, prefix = -math.inf, math.inf
        for j in _window(i, n, phi.bound):
            best = max(best, min(robustness(phi.right, trace, j), prefix))
            prefix = min(prefix, robustness(phi.left, trace, j))
        return best
    raise TypeError(f"unknown formula {phi!r}")


# ------------------------------------------------------------------------------------ CTL helpers

def _paths(outcomes: Any) -> torch.Tensor:
    """``[W, Π]`` (victory on election day) or ``[W, Π, D]`` (atom per depth) → ``[W, Π, D]`` Boolean."""
    t = torch.as_tensor(outcomes)
    if t.dtype != torch.bool:
        raise TypeError("outcomes must be Boolean")
    if t.dim() == 2:
        t = t.unsqueeze(-1)
    if t.dim() != 3:
        raise ValueError("outcomes must be [W, Π] or [W, Π, D]")
    return t


def exists_eventually(outcomes: Any) -> bool:
    """CTL ∃◇φ over the worlds × policies tree: some path (w, π) reaches φ at some depth."""
    return bool(_paths(outcomes).any())


def forall_eventually(outcomes: Any) -> bool:
    """CTL ∀◇φ: every path (every world, every plan) reaches φ at some depth."""
    return bool(_paths(outcomes).any(dim=2).all())


def strategy_eventually(outcomes: Any) -> bool:
    """Strategic ∃π ∀w ◇φ (ATL-style ⟨⟨p⟩⟩◇φ restricted to Π): one plan reaches φ in every world."""
    return bool(_paths(outcomes).any(dim=2).all(dim=0).any())


# ---------------------------------------------------------------------------------- game traces

def game_trace(observation: Mapping[str, Any], timeline: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The player's per-round trace (module docstring) from a final observation (+ optional timeline).

    States are rounds 1..T (decision points) followed by the election-day state. Only player-visible
    data (polls, treasury, result) and the timeline's knowledge booleans are used.
    """
    from .knowledge import observation_at

    rounds = int(observation["rounds"])
    polls = {int(p["round"]): float(p["estimate"]) for p in observation.get("polls") or []}
    by_round = {int(r["round"]): r for r in (timeline or {}).get("rounds", [])}
    trace: list[dict[str, Any]] = []
    for t in range(1, rounds + 1):
        if t not in polls:
            break
        obs_t = observation_at(dict(observation), t, stage="decision")
        row = by_round.get(t, {})
        trace.append({"round": t, "Lead": polls[t] > 0.5, "margin": polls[t] - 0.5,
                      "BudgetLeft": Decimal(str(obs_t["treasury"])) > 0, "Victory": False,
                      "Viable": bool(row.get("possible_victory", False)),
                      "Guaranteed": bool(row.get("guaranteed_victory", False))})
    result = observation.get("result")
    if result is not None and trace:
        share = float(result["vote_share"])
        trace.append({"round": rounds + 1, "Lead": share > 0.5, "margin": share - 0.5,
                      "BudgetLeft": Decimal(str(observation["treasury"])) > 0, "Victory": bool(result["victory"]),
                      "Viable": bool(result["victory"]), "Guaranteed": bool(result["victory"])})
    return trace
