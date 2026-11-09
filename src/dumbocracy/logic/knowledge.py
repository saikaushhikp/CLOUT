"""The player's knowledge log K_p at an earlier decision point (spec §6.5, §10.1; P5).

The epistemic timeline (web_api_v2.md "Epistemic timeline (P5)") asks, for every round t, what the
player knew *at that moment*. The final contract ``Observation`` already holds the whole knowledge log
with round tags, so the observation at an earlier time is a pure filter of it — no hidden state is read.

Two decision points per round t (1 ≤ t ≤ T):

* ``stage="start"`` — the START of round t, right after that round's free statistic and poll were
  added and before the player acts in round t (this is exactly what the engine shows at that moment);
* ``stage="decision"`` — the moment the round-t allocation is chosen: as ``"start"`` plus every query
  the player asked during round t (and its cost). The timeline uses this point, because that is the
  knowledge the round-t plan was based on.

Filters (round tags as written by the engine: SURVEY queries carry round 0, round-t items round t, the
pre-election poll round T+1): ``free_stats`` round ≤ t; ``polls`` round ≤ t; ``queries`` asked in SURVEY or
in a round < t (``"decision"``: ≤ t); ``allocations`` round < t. The treasury is recomputed from the
costs: initial budget (final treasury + every cost in the log) − included query and allocation costs;
``media_airings`` from the included allocations; ``round_reports`` round < t; ``workers`` (R16) with
``hired`` = Σ ``extra_workers`` of the included allocations and ``available`` = free + hired;
``phase`` "ROUND", ``round`` t, ``result`` None,
``legal_actions`` as the engine lists them at that point; ``events`` keep rounds < t (approximate —
the event log is not part of the knowledge log).
"""
from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

from ..sectors import NUM_REGIONS, REGIONS

__all__ = ["decision_rounds", "initial_budget", "observation_at"]

_STAGES = ("start", "decision")


def _money(value: Any) -> Decimal:
    return Decimal(str(value))


def _spent(items: list[dict[str, Any]]) -> Decimal:
    return sum((_money(x.get("cost", "0")) for x in items), Decimal("0.00"))


def initial_budget(observation: dict[str, Any]) -> Decimal:
    """Starting treasury: the observation's ``initial_budget`` (R17) if present, else the current treasury
    + every logged cost."""
    if observation.get("initial_budget") is not None:
        return _money(observation["initial_budget"]).quantize(Decimal("0.01"))
    return (_money(observation["treasury"]) + _spent(list(observation.get("queries") or []))
            + _spent(list(observation.get("allocations") or []))).quantize(Decimal("0.01"))


def decision_rounds(observation: dict[str, Any]) -> list[int]:
    """Rounds t = 1..T that have been reached (a poll for round t exists), in order."""
    polled = {int(p["round"]) for p in observation.get("polls") or []}
    return [t for t in range(1, int(observation["rounds"]) + 1) if t in polled]


def _query_included(q: dict[str, Any], t: int, stage: str) -> bool:
    if q.get("phase") == "SURVEY":
        return True
    r = int(q.get("round", 0))
    return r < t or (stage == "decision" and r == t)


def observation_at(observation: dict[str, Any], round_index: int, *, stage: str = "start") -> dict[str, Any]:
    """The player's observation at the start (or decision point) of round ``round_index`` (module docstring).

    ``observation`` is any later contract ``Observation`` of the same player (typically the final one at
    DONE). Returns a new dict; the input is not modified.
    """
    if stage not in _STAGES:
        raise ValueError(f"stage must be one of {_STAGES}")
    t = int(round_index)
    rounds = int(observation["rounds"])
    if not 1 <= t <= rounds:
        raise ValueError(f"round_index must lie in 1..{rounds}")
    if observation.get("manifesto") is None:
        raise ValueError("no manifesto committed: the game never reached round 1")
    budget = initial_budget(observation)
    out = copy.deepcopy(observation)
    out["free_stats"] = [e for e in out.get("free_stats") or [] if int(e["round"]) <= t]
    out["polls"] = [p for p in out.get("polls") or [] if int(p["round"]) <= t]
    out["queries"] = [q for q in out.get("queries") or [] if _query_included(q, t, stage)]
    out["allocations"] = [a for a in out.get("allocations") or [] if int(a["round"]) < t]
    treasury = budget - _spent(out["queries"]) - _spent(out["allocations"])
    out["treasury"] = f"{treasury.quantize(Decimal('0.01')):.2f}"
    airings = [0] * NUM_REGIONS
    for a in out["allocations"]:
        airings = [x + int(bool(m)) for x, m in zip(airings, a["media_regions"])]
    out["media_airings"] = dict(zip(REGIONS, airings))
    if "round_reports" in out:
        out["round_reports"] = [r for r in out["round_reports"] or [] if int(r["round"]) < t]
    hired = sum(int(a.get("extra_workers", 0)) for a in out["allocations"])
    free = int((out.get("workers") or {}).get("free", out.get("costs", {}).get("free_workers", 0)))
    out["workers"] = {"free": free, "hired": hired, "available": free + hired}
    out["phase"] = "ROUND"
    out["round"] = t
    out["result"] = None
    query_cost = _money(out.get("costs", {}).get("query", "5.00"))
    out["legal_actions"] = (["query"] if treasury >= query_cost else []) + ["allocate", "forfeit"]
    if "events" in out:
        out["events"] = [e for e in out["events"] if int(e.get("round", 0)) < t]
    return out
