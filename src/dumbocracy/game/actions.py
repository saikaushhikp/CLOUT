"""Player actions and PURE validators with stable error codes (spec §7; web_api_v2.md "Errors").

Validators never mutate anything: they return ``None`` (valid) or a :class:`Rejection`. The engine
applies an action only after its validator passed, so rejected actions change nothing (atomic).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Union

import torch

from ..dynamics.campaign import eligible_counts, eligible_workers, media_cost, worker_cost
from ..sectors import NUM_REGIONS, NUM_SECTORS, REGIONS, SECTORS

if TYPE_CHECKING:  # pragma: no cover
    from .state import GameState, PlayerState


class ErrorCode:
    """Stable rejection codes (contract + spec §7)."""

    INVALID_ACTION = "INVALID_ACTION"
    INVALID_PHASE = "INVALID_PHASE"
    INVALID_MANIFESTO = "INVALID_MANIFESTO"
    INVALID_ALLOCATION = "INVALID_ALLOCATION"
    INVALID_QUERY = "INVALID_QUERY"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    UNKNOWN_PLAYER = "UNKNOWN_PLAYER"
    TERMINAL_STATE = "TERMINAL_STATE"


MANIFESTO_TOLERANCE = 1e-6  # |Σb − 1| (spec §7)


@dataclass(frozen=True, slots=True)
class Rejection:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Query:
    """Paid sampled-survey query (spec §6.3, R12): % of a fresh 10% sample of ``region`` voters with
    v[sector_b] > v[sector_a]. Repeats are allowed (each draws a new sample); (b, a, r) is a different key."""

    sector_a: str
    sector_b: str
    region: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.sector_a, self.sector_b, self.region)


@dataclass(frozen=True, slots=True)
class Commit:
    """Fix the manifesto b ∈ Δ (spec §7)."""

    manifesto: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Allocate:
    """One round's campaign allocation (P8: NEW workers bought per region + media regions)."""

    hires_by_region: tuple[int, ...] = (0,) * NUM_REGIONS
    media_regions: tuple[bool, ...] = (False,) * NUM_REGIONS


@dataclass(frozen=True, slots=True)
class Forfeit:
    """Timer expired: equivalent to allocating nothing (spec §7)."""


Action = Union[Query, Commit, Allocate, Forfeit]
ACTION_TYPES: dict[str, type] = {"query": Query, "commit": Commit, "allocate": Allocate, "forfeit": Forfeit}


def action_type(action: Action) -> str:
    return {Query: "query", Commit: "commit", Allocate: "allocate", Forfeit: "forfeit"}[type(action)]


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Parse an action dict (``{"type": "query", ...}``) as produced by a :class:`Player`.

    Raises ``ValueError`` on an unknown type or missing/extra fields; value validation is left to
    the validators so that it yields the stable error codes.
    """
    if not isinstance(data, Mapping) or data.get("type") not in ACTION_TYPES:
        raise ValueError(f"unknown action type {data.get('type') if isinstance(data, Mapping) else data!r}")
    kind = data["type"]
    fields = {k: v for k, v in data.items() if k != "type"}
    if kind == "commit":
        raw = fields["manifesto"]
        return Commit(tuple(raw) if isinstance(raw, (list, tuple)) else raw)
    if kind == "allocate":
        unknown = set(fields) - {"hires_by_region", "media_regions"}
        if unknown:
            raise ValueError(f"unknown allocate fields {sorted(unknown)}")
        hires = fields["hires_by_region"]
        media = fields.get("media_regions", (False,) * NUM_REGIONS)
        return Allocate(tuple(hires) if isinstance(hires, (list, tuple)) else hires,
                        tuple(media) if isinstance(media, (list, tuple)) else media)
    return ACTION_TYPES[kind](**fields)


def action_to_dict(action: Action) -> dict[str, Any]:
    """JSON-safe inverse of :func:`action_from_dict`."""
    if isinstance(action, Query):
        return {"type": "query", "sector_a": action.sector_a, "sector_b": action.sector_b, "region": action.region}
    if isinstance(action, Commit):
        return {"type": "commit", "manifesto": [float(x) for x in action.manifesto]}
    if isinstance(action, Allocate):
        return {"type": "allocate", "hires_by_region": list(action.hires_by_region),
                "media_regions": list(action.media_regions)}
    return {"type": "forfeit"}


# ----------------------------------------------------------------------------------- helpers

def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def allocation_cost(action: Allocate, sizes: Sequence[int], unit_worker_cost: Decimal, media_base_cost: Decimal) -> Decimal:
    """``Σ hires × worker_cost + Σ media price over aired regions`` (P8, spec §3.2), exact Decimal."""
    return worker_cost(int(sum(action.hires_by_region)), unit_worker_cost) + \
        media_cost(list(sizes), list(action.media_regions), media_base_cost)


def available_hires(state: "GameState", player: "PlayerState") -> list[int]:
    """Voters per region that ``player`` could still buy as workers this round (R22): degree ≥ min_degree,
    not a zealot, not anyone's worker, minus other players' pending hires this round."""
    e = state.electorate
    taken = torch.zeros(e.num_voters, dtype=torch.bool, device=e.device)
    for p in state.players:
        if p.workers is not None:
            taken |= p.workers.pinned
    counts = eligible_counts(e.region_of, eligible_workers(e.graph.degree, e.susceptibility.zealot, taken,
                                                           state.config.workers.min_degree))
    for p in state.players:
        if p.player_id != player.player_id and p.pending is not None:
            counts = [c - int(h) for c, h in zip(counts, p.pending.hires_by_region)]
    return counts


def _phase_guard(state: "GameState", expected: tuple[str, ...]) -> Rejection | None:
    if state.phase == "DONE":
        return Rejection(ErrorCode.TERMINAL_STATE, "the game is over")
    if state.phase not in expected:
        return Rejection(ErrorCode.INVALID_PHASE, f"not allowed in phase {state.phase}")
    return None


# -------------------------------------------------------------------------------- validators

def validate_query(state: "GameState", player: "PlayerState", action: Query) -> Rejection | None:
    """Spec §6.3 (R12): SURVEY or ROUND; distinct known sectors, known region; affordable.
    Repeating a query is allowed (no duplicate rejection since R12)."""
    rejection = _phase_guard(state, ("SURVEY", "ROUND"))
    if rejection:
        return rejection
    if action.sector_a not in SECTORS or action.sector_b not in SECTORS:
        return Rejection(ErrorCode.INVALID_QUERY, f"sectors must be in {list(SECTORS)}")
    if action.region not in REGIONS:
        return Rejection(ErrorCode.INVALID_QUERY, f"region must be in {list(REGIONS)}")
    if action.sector_a == action.sector_b:
        return Rejection(ErrorCode.INVALID_QUERY, "sector_a and sector_b must differ")
    if not player.ledger.can_afford(state.config.survey.query_cost):
        return Rejection(ErrorCode.INSUFFICIENT_FUNDS, "not enough money for a query")
    return None


def validate_commit(state: "GameState", player: "PlayerState", action: Commit) -> Rejection | None:
    """Spec §7: SURVEY only, once; b has 5 finite entries ≥ 0 summing to 1 ± 1e-6."""
    rejection = _phase_guard(state, ("SURVEY",))
    if rejection:
        return rejection
    if player.manifesto is not None:
        return Rejection(ErrorCode.INVALID_PHASE, "manifesto already committed")
    b = action.manifesto
    if not isinstance(b, tuple) or len(b) != NUM_SECTORS or not all(_is_number(x) for x in b):
        return Rejection(ErrorCode.INVALID_MANIFESTO, f"manifesto must be {NUM_SECTORS} finite numbers")
    if any(x < 0 for x in b):
        return Rejection(ErrorCode.INVALID_MANIFESTO, "manifesto entries must be >= 0")
    if abs(math.fsum(b) - 1.0) > MANIFESTO_TOLERANCE:
        return Rejection(ErrorCode.INVALID_MANIFESTO, "manifesto must sum to 1 (± 1e-6)")
    return None


def _round_guard(state: "GameState", player: "PlayerState") -> Rejection | None:
    rejection = _phase_guard(state, ("ROUND",))
    if rejection:
        return rejection
    if player.ready:
        return Rejection(ErrorCode.INVALID_PHASE, "already allocated this round; waiting for other players")
    return None


def validate_allocate(state: "GameState", player: "PlayerState", action: Allocate) -> Rejection | None:
    """P8 + contract: 4 non-negative int ``hires_by_region`` (each region must have that many eligible
    voters, R22), 4 boolean media flags; cost (hires × worker fee + media) ≤ treasury."""
    rejection = _round_guard(state, player)
    if rejection:
        return rejection
    hires, media = action.hires_by_region, action.media_regions
    if not isinstance(hires, tuple) or len(hires) != NUM_REGIONS or not all(_is_int(h) and h >= 0 for h in hires):
        return Rejection(ErrorCode.INVALID_ALLOCATION, "hires_by_region must be 4 non-negative integers")
    if not isinstance(media, tuple) or len(media) != NUM_REGIONS or not all(isinstance(m, bool) for m in media):
        return Rejection(ErrorCode.INVALID_ALLOCATION, "media_regions must be 4 booleans")
    if any(hires):
        available = available_hires(state, player)
        for r, (h, a) in enumerate(zip(hires, available)):
            if h > a:
                return Rejection(ErrorCode.INVALID_ALLOCATION,
                                 f"region {REGIONS[r]} has only {a} voters eligible as workers, {h} requested")
    cost = allocation_cost(action, state.sizes, state.config.workers.worker_cost, state.config.media.base_cost)
    if not player.ledger.can_afford(cost):
        return Rejection(ErrorCode.INSUFFICIENT_FUNDS, f"allocation costs {cost:.2f} but treasury is {player.ledger.balance:.2f}")
    return None


def validate_forfeit(state: "GameState", player: "PlayerState", action: Forfeit) -> Rejection | None:
    """Spec §7: ROUND only, player not yet ready."""
    return _round_guard(state, player)


VALIDATORS = {Query: validate_query, Commit: validate_commit, Allocate: validate_allocate, Forfeit: validate_forfeit}


def validate(state: "GameState", player: "PlayerState", action: object) -> Rejection | None:
    """Dispatch to the pure validator for ``action``'s type."""
    validator = VALIDATORS.get(type(action))
    if validator is None:
        return Rejection(ErrorCode.INVALID_ACTION, f"unknown action {type(action).__name__}")
    return validator(state, player, action)
