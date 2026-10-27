"""Join completed-trace diagnostics with optional scoped finite-world results."""
from __future__ import annotations

from typing import Iterable

from .engine import GameEngine, GameState
from .epistemic import check_finite_worlds
from .logic import CanonicalTrace, evaluate_diagnostics


def verify_episode(state: GameState, engine: GameEngine | None = None,
                   worlds: Iterable[GameState] | None = None, *,
                   node_limit: int = 10000) -> dict[str, object]:
    """Return trace-only LTL and, when supplied, finite-model epistemic results.

    Missing worlds are explicitly reported as not supplied, never inferred from
    one completed trace. The engine only drives the finite-world checker.
    """
    trace = CanonicalTrace.from_game_state(state)
    report: dict[str, object] = {
        "trace": {
            "evaluation_mode": "trace_only",
            "scope": "the supplied completed fixed-horizon episode only",
            "horizon": trace.horizon,
            "diagnostics": {name: result.to_public_dict()
                            for name, result in evaluate_diagnostics(trace).items()},
        },
        "finite_model": {
            "evaluation_mode": "finite_model",
            "status": "NOT_SUPPLIED",
            "scope": "no finite world set supplied; no branching claim made",
        },
    }
    if worlds is not None:
        if engine is None:
            raise ValueError("engine is required when finite worlds are supplied")
        world_tuple = tuple(worlds)
        report["finite_model"] = check_finite_worlds(
            engine, world_tuple, node_limit=node_limit
        ).to_public_dict()
    return report
