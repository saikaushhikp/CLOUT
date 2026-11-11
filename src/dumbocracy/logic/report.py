"""Post-game epistemic timeline for the web debrief and the research layer (R14; web_api_v2.md "Epistemic timeline (P5)").

For each decision point t = 1..T (round t, after that round's poll and the player's round-t queries —
:func:`.knowledge.observation_at` with ``stage="decision"``) the timeline evaluates, over a finite sample
W_t of worlds consistent with what the player knew (:func:`.worlds.consistent_worlds`) and the finite plan
set Π_t (:func:`.epistemic.policy_set`), the knowledge properties of :mod:`.epistemic`.

Separation of knowledge and truth: every epistemic field (possible/guaranteed/knows/inevitable,
best_plan, knowledge stats) is computed from the reconstructed player observation only. The single
ground-truth field ``truth_victory_possible`` evaluates the same Π_t in the TRUE world at that moment,
obtained by replaying the recorded actions through the engine (same electorate, same seeds). Only
aggregated Booleans/shares leave this module — no σ, τ or per-voter data.

Deterministic per ``seed``: every random draw (worlds, campaign randomness in rollouts) is keyed from it.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import torch

from ..config import SeedStreams
from ..game.actions import Allocate, Forfeit
from ..game.engine import GameEngine
from ..game.observation import observation_for
from ..strategies.players import public_config
from .epistemic import evaluate, policy_set, properties, truth_victory_possible
from .knowledge import decision_rounds, observation_at
from .worlds import consistent_worlds

__all__ = ["decision_states", "epistemic_timeline", "headline", "timeline_with_diagnostics"]

SCOPE_NOTE = ("Knowledge is evaluated over a finite sample of {w} worlds per round that are consistent with what "
              "you knew then (your surveys, the free statistics, the polls and your own actions; synthetic "
              "{n}-voter worlds) and a finite set of up to {p} campaign plans for the remaining rounds. "
              "'Guaranteed' means one plan wins in every sampled world, not a proof over all possible worlds. "
              "Victory = expected vote share above 50%.")


def decision_states(state, player_id: str) -> dict[int, Any]:
    """TRUE engine states just before ``player_id`` allocated (or forfeited) in each round (hidden; truth only).

    Rebuilt by re-applying the recorded accepted actions to a fresh game on the same electorate
    (the engine is deterministic per (config, electorate)), so no per-round snapshot needs to be stored.
    """
    current = GameEngine.new_game(state.config, state.player_ids, game_id=state.game_id, electorate=state.electorate)
    out: dict[int, Any] = {}
    for pid, action in state.history:
        if (pid == player_id and isinstance(action, (Allocate, Forfeit)) and current.phase == "ROUND"
                and current.round not in out):
            out[current.round] = current
        step = GameEngine.apply(current, pid, action)
        if not step.accepted:
            raise RuntimeError(f"history replay rejected {action!r}: {step.error_code}")
        current = step.state
    return out


def _first(rows: list[dict[str, Any]], key: str, value: bool = True) -> int | None:
    for row in rows:
        if bool(row[key]) is value:
            return int(row["round"])
    return None


def headline(rows: list[dict[str, Any]], actual: dict[str, Any]) -> str:
    """One or two plain sentences summarising the timeline (derived only from the rows and the result)."""
    if not rows:
        return "No campaign rounds were played, so there is no timeline."
    rounds = [int(r["round"]) for r in rows]
    possible_all = all(r["possible_victory"] for r in rows)
    first_known = _first(rows, "guaranteed_victory")
    first_that = _first(rows, "knows_win_exists")
    lost_hope = _first(rows, "possible_victory", False)
    parts: list[str] = []
    if possible_all:
        parts.append("Victory looked possible throughout")
    elif lost_hope == rounds[0]:
        parts.append("From round 1 no plan won in any world you considered possible")
    else:
        parts.append(f"Victory looked possible until round {lost_hope - 1}, then no plan won in any world you considered possible")
    if first_known is not None:
        parts[-1] += f"; you knew a winning plan from round {first_known}."
    elif first_that is not None:
        parts[-1] += f"; from round {first_that} you knew some plan would win, but not which one."
    elif possible_all:
        parts[-1] += ", but you never knew a plan that would win in every world you considered possible."
    else:
        parts[-1] += "."
    # Truth vs knowledge (truth ≠ knowledge): cite when winning really became impossible, separately from the
    # rounds in which the player's knowledge disagreed with the truth.
    def _rounds(items: list[int]) -> str:
        return f"round {items[0]}" if len(items) == 1 else "rounds " + ", ".join(map(str, items[:-1])) + f" and {items[-1]}"
    false_hope = [int(r["round"]) for r in rows if r["possible_victory"] and not r["truth_victory_possible"]]
    missed = [int(r["round"]) for r in rows if r["truth_victory_possible"] and not r["possible_victory"]]
    if false_hope:
        truth_lost = _first(rows, "truth_victory_possible", False)
        parts.append(f"In reality no plan could win from round {truth_lost} on, yet you still saw a chance in {_rounds(false_hope)}.")
    if missed:
        parts.append(f"In reality a winning plan still existed in {_rounds(missed)} even though you could not see one.")
    outcome = "won" if actual["victory"] else "lost"
    parts.append(f"You {outcome} with {100.0 * float(actual['vote_share']):.1f}% of the vote.")
    return " ".join(parts)


def epistemic_timeline(state, player_id: str, *, worlds: int = 16, world_voters: int = 1600, seed: int = 0,
                       device=None, max_policies: int | None = None) -> dict:
    """``EpistemicTimeline`` (web_api_v2.md) for ``player_id`` of a finished game (raises ``ValueError``
    before DONE and ``KeyError`` for an unknown player). ``device`` is used for the synthetic worlds
    (default CPU); the true-world rollouts run on the electorate's own device."""
    return timeline_with_diagnostics(state, player_id, worlds=worlds, world_voters=world_voters, seed=seed,
                                     device=device, max_policies=max_policies)[0]


def timeline_with_diagnostics(state, player_id: str, *, worlds: int = 16, world_voters: int = 1600, seed: int = 0,
                              device=None, max_policies: int | None = None) -> tuple[dict, dict[str, Any]]:
    """:func:`epistemic_timeline` plus research diagnostics (timing, per-round draws/acceptance/fallback,
    per-round plan win fractions and true-world shares). The diagnostics are NOT for players.
    ``max_policies`` caps |Π_t| (:func:`.epistemic.policy_set`; default ``None`` = all candidates)."""
    if not state.has_player(player_id):
        raise KeyError(player_id)
    if state.phase != "DONE":
        raise ValueError("the epistemic timeline is only available once the game is DONE")
    start = time.perf_counter()
    observation = observation_for(state, player_id)
    config = public_config(observation, replace(state.config, device="cpu"))
    streams = SeedStreams(int(seed))
    world_seed = streams.derive("worlds", "logic-timeline", player_id)
    truth_states = decision_states(state, player_id)
    rows: list[dict[str, Any]] = []
    largest_plan_set = 0
    diagnostics = []
    for t in decision_rounds(observation):
        obs_t = observation_at(observation, t, stage="decision")
        ws = consistent_worlds(obs_t, config, worlds, world_voters=world_voters, seed=world_seed, device=device)
        plans = policy_set(obs_t, config, max_policies)
        largest_plan_set = max(largest_plan_set, len(plans))
        eval_seed = streams.derive("worlds", "logic-eval", player_id, t)
        values = evaluate(ws, plans, obs_t, config, seed=eval_seed)
        props = properties(values, [p.label for p in plans])
        truth = truth_victory_possible(truth_states[t], plans, player_id=player_id, observation_t=obs_t, seed=eval_seed) \
            if t in truth_states else False
        rows.append({
            "round": t,
            "treasury": str(obs_t["treasury"]),
            "knowledge": {"queries": len(obs_t.get("queries") or []), "polls": len(obs_t.get("polls") or []),
                          "posterior_entropy": float(ws.posterior.entropy()), "consistent_worlds": len(ws)},
            "possible_victory": props["possible_victory"],
            "guaranteed_victory": props["guaranteed_victory"],
            "knows_win_exists": props["knows_win_exists"],
            "inevitable_defeat": props["inevitable_defeat"],
            "truth_victory_possible": bool(truth),
            "best_plan": props["best_plan"],
        })
        diagnostics.append({"round": t, "draws": ws.draws, "accepted": ws.accepted, "fallback": ws.fallback,
                            "num_policies": len(plans), "guaranteed_witness": props["guaranteed_witness"]})
    result = observation["result"] or {}
    actual = {"victory": bool(result.get("victory", False)), "vote_share": float(result.get("vote_share", 0.0)),
              "expected_share": float(result.get("expected_share", 0.0))}
    n_world = min(int(world_voters), int(observation["num_voters"]))
    out = {
        "scope": {"worlds_per_round": int(worlds), "policies_per_round": int(largest_plan_set), "world_voters": n_world,
                  "note": SCOPE_NOTE.format(w=int(worlds), n=n_world, p=int(largest_plan_set))},
        "rounds": rows,
        "actual": actual,
        "headline": headline(rows, actual),
    }
    return out, {"seconds": time.perf_counter() - start, "rounds": diagnostics}
