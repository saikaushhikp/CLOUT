"""Immutable, player-keyed game-state snapshots (spec §8).

A new :class:`GameState` is produced for every accepted action; rejected actions return the very
same object. Tensors held here (electorate, preferences, alignment snapshots) are never mutated in
place, so snapshots can share them safely.

Hidden fields (never serialised to a player; see :mod:`.observation`): ``electorate``,
``preferences``, ``PlayerState.share_trajectory``, ``PlayerState.alignment``,
``PlayerState.pending``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from ..config import GameConfig
from ..electorate import Electorate
from ..information.knowledge import KnowledgeLog
from ..voting import ElectionResult
from .actions import Action, Allocate
from .ledger import Ledger

Phase = Literal["SURVEY", "ROUND", "DONE"]


@dataclass(frozen=True, slots=True)
class Event:
    """Public event-log entry; ``audience`` = None (everyone) or a single player id."""

    seq: int
    round: int
    type: str
    message: str
    audience: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "round": self.round, "type": self.type, "message": self.message}


@dataclass(frozen=True, eq=False)
class PlayerState:
    """One player's slice of the game (spec §8)."""

    player_id: str
    ledger: Ledger
    manifesto: tuple[float, ...] | None = None
    knowledge: KnowledgeLog = field(default_factory=KnowledgeLog)
    allocations: tuple[Allocate, ...] = ()             # resolved + pending, one per round
    media_airings: tuple[int, int, int, int] = (0, 0, 0, 0)
    hired_workers: int = 0                              # R16: permanent hires so far (incl. a pending round's)
    ready: bool = False                                 # committed (SURVEY) / allocated (ROUND)
    pending: Allocate | None = None                     # this round's allocation awaiting resolution
    query_count: int = 0                                # paid queries asked so far in the game (survey-stream key)
    # R12 pooling accumulators for THIS phase/round: (sector_a, sector_b, region) -> (Σyes, Σm, repeats).
    # Replaced (never mutated) on every query; reset when a new round starts.
    query_pools: dict[tuple[str, str, str], tuple[int, int, int]] = field(default_factory=dict)
    result: ElectionResult | None = None
    share_trajectory: tuple[float, ...] = ()            # HIDDEN: true E before round 1 and after each round
    alignment: tuple[torch.Tensor, ...] = ()            # HIDDEN: uint8 [N] similarity bins per round 0..T (CPU)


@dataclass(frozen=True, eq=False)
class GameState:
    """Snapshot of one game (spec §8). ``players`` is ordered like ``player_ids``."""

    game_id: str
    config: GameConfig
    electorate: Electorate                  # HIDDEN, shared, never mutated
    census: dict[str, Any]                  # public structural census (§6.1), computed once
    preferences: torch.Tensor               # HIDDEN v(t) [N, 5]
    phase: Phase
    round: int                              # 0 in SURVEY; 1..T planning; T at DONE
    player_ids: tuple[str, ...]
    players: tuple[PlayerState, ...]
    events: tuple[Event, ...] = ()
    history: tuple[tuple[str, Action], ...] = ()   # accepted actions, for replay (spec §8 trace)

    @property
    def device(self) -> torch.device:
        return self.electorate.device

    @property
    def sizes(self) -> list[int]:
        return [int(x) for x in self.electorate.sizes.tolist()]

    def has_player(self, player_id: str) -> bool:
        return player_id in self.player_ids

    def player(self, player_id: str) -> PlayerState:
        return self.players[self.player_ids.index(player_id)]

    def with_player(self, updated: PlayerState) -> "GameState":
        index = self.player_ids.index(updated.player_id)
        players = self.players[:index] + (updated,) + self.players[index + 1:]
        return replace(self, players=players)

    def with_event(self, type: str, message: str, audience: str | None = None) -> "GameState":
        event = Event(len(self.events) + 1, self.round, type, message, audience)
        return replace(self, events=self.events + (event,))
