"""Allowlisted, JSON-safe per-player observation (spec §6.1–§6.6, §8; web_api_v2.md ``Observation``).

Built ONLY from public fields: config costs/rules, the public census, the public libraries and the
player's own knowledge log / ledger / allocations. Nothing hidden (per-voter values, σ/τ, zealots,
p_r/p_rs/s, edges, true expected shares during play) is ever read into the payload.
"""
from __future__ import annotations

import copy
from typing import Any

from ..dynamics.campaign import media_cost
from ..sectors import ARCHETYPES, PERSONALITIES, REGION_LABELS, REGIONS, SECTOR_LABELS, SECTORS
from .engine import public_result
from .ledger import money_str
from .state import GameState, PlayerState

SCHEMA_VERSION = "2.2"


def _beta_mean(params: tuple[float, float]) -> float:
    a, b = params
    return a / (a + b)


def _costs(state: GameState) -> dict[str, Any]:
    cfg = state.config
    sizes = state.sizes
    media = {r: money_str(media_cost(sizes, [i == j for j in range(len(REGIONS))], cfg.media.base_cost))
             for i, r in enumerate(REGIONS)}
    return {"query": money_str(cfg.survey.query_cost), "worker": money_str(cfg.workers.worker_cost),
            "min_degree": cfg.workers.min_degree, "attempt_cap": cfg.workers.attempt_cap,
            "media_by_region": media}


def _regions(state: GameState) -> list[dict[str, Any]]:
    census = state.census["regions"]
    return [{"id": row["region"], "label": REGION_LABELS[i], "voters": row["voters"], "intra_edges": row["intra_edges"],
             "mean_degree": row["mean_degree"]} for i, row in enumerate(census)]


def _libraries() -> dict[str, Any]:
    return {
        "sectors": [{"id": s, "label": label} for s, label in zip(SECTORS, SECTOR_LABELS)],
        "archetype_library": [{"id": a.id, "name": a.name, "mean": list(a.mean)} for a in ARCHETYPES],
        "personality_library": [{"id": p.id, "name": p.name, "social": _beta_mean(p.alpha),
                                 "media": _beta_mean(p.beta), "campaign": _beta_mean(p.gamma)} for p in PERSONALITIES],
    }


def _legal_actions(state: GameState, player: PlayerState) -> list[str]:
    if state.phase == "DONE":
        return []
    actions = ["query"] if player.ledger.can_afford(state.config.survey.query_cost) else []
    if state.phase == "SURVEY" and player.manifesto is None:
        actions.append("commit")
    if state.phase == "ROUND" and not player.ready:
        actions += ["allocate", "forfeit"]
    return actions


def _from_knowledge(player: PlayerState) -> dict[str, Any]:
    log = player.knowledge
    return {
        "free_stats": [{"round": e.round, "top3_pct": e.payload["top3_pct"]} for e in log.of_kind("free_stat")],
        "queries": [e.payload for e in log.of_kind("query")],
        "polls": [{"round": e.round, **e.payload} for e in log.of_kind("poll")],
        "allocations": [{"round": e.round, **e.payload} for e in log.of_kind("allocation")],
        "round_reports": [dict(e.payload) for e in log.of_kind("round_report")],   # R19, oldest first
    }


def _workers(state: GameState, player: PlayerState) -> dict[str, Any]:
    """P8: the permanent team per region (counts only; never node ids), incl. an accepted round's hires."""
    team = [int(x) for x in player.team_by_region]
    return {"by_region": dict(zip(REGIONS, team)), "total": sum(team)}


def observation_for(state: GameState, player_id: str) -> dict[str, Any]:
    """Contract ``Observation`` for ``player_id`` (raises ``KeyError`` for an unknown player)."""
    if not state.has_player(player_id):
        raise KeyError(player_id)
    player = state.player(player_id)
    cfg = state.config
    census = state.census
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": state.game_id,
        "player_id": player_id,
        "phase": state.phase,
        "round": state.round,
        "rounds": cfg.rounds,
        "num_voters": int(census["num_voters"]),
        "seed": cfg.seed,
        "treasury": money_str(player.ledger.balance),
        "initial_budget": money_str(cfg.budget),
        "workers": _workers(state, player),
        "costs": _costs(state),
        **_libraries(),
        "regions": _regions(state),
        "inter_edges": copy.deepcopy(census["inter_edges"]),
        **_from_knowledge(player),
        "manifesto": None if player.manifesto is None else list(player.manifesto),
        "media_airings": dict(zip(REGIONS, player.media_airings)),
        "legal_actions": _legal_actions(state, player),
        "result": public_result(player.result) if state.phase == "DONE" else None,
        "events": [e.to_public_dict() for e in state.events if e.audience in (None, player_id)],
    }
