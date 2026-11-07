"""Per-player knowledge log K_p (spec §6.5): immutable, append-only, JSON-safe.

Every observation item (free statistics, query answers, polls, own actions and their public
outcomes) is appended as a :class:`KnowledgeEntry`. The epistemic layer (P5) replays candidate
worlds against this log to build the information set I(K_p). Payloads hold only JSON primitives
(str/int/float/bool/None) and nested lists/dicts of them; they are frozen into tuples internally
so the log can be shared between immutable game-state snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

KnowledgeKind = Literal["free_stat", "query", "poll", "commit", "allocation", "round_report", "result"]
KINDS: tuple[str, ...] = ("free_stat", "query", "poll", "commit", "allocation", "round_report", "result")

_Frozen = Any


def _freeze(value: Any) -> _Frozen:
    """Convert JSON-like data to a hashable, immutable structure (dict -> ("__dict__", pairs))."""
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("knowledge payload keys must be strings")
        return ("__dict__", tuple((k, _freeze(item)) for k, item in value.items()))
    if isinstance(value, (list, tuple)):
        return ("__list__", tuple(_freeze(item) for item in value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"knowledge payload values must be JSON primitives, got {type(value).__name__}")


def _thaw(value: _Frozen) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__dict__":
        return {k: _thaw(v) for k, v in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__list__":
        return [_thaw(item) for item in value[1]]
    return value


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """One observed fact: ``kind`` at ``round`` (0 = survey; T+1 = pre-election) with a payload."""

    kind: str
    round: int
    frozen_payload: _Frozen

    @classmethod
    def make(cls, kind: KnowledgeKind, round: int, payload: dict[str, Any]) -> "KnowledgeEntry":
        if kind not in KINDS:
            raise ValueError(f"unknown knowledge kind {kind!r}")
        return cls(kind, int(round), _freeze(dict(payload)))

    @property
    def payload(self) -> dict[str, Any]:
        """A fresh, mutable copy of the payload (callers cannot alter the log)."""
        return _thaw(self.frozen_payload)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "round": self.round, "payload": self.payload}


@dataclass(frozen=True, slots=True)
class KnowledgeLog:
    """Append-only sequence of :class:`KnowledgeEntry`; ``append`` returns a new log."""

    entries: tuple[KnowledgeEntry, ...] = field(default_factory=tuple)

    def append(self, entry: KnowledgeEntry) -> "KnowledgeLog":
        return KnowledgeLog(self.entries + (entry,))

    def extend(self, entries: tuple[KnowledgeEntry, ...] | list[KnowledgeEntry]) -> "KnowledgeLog":
        return KnowledgeLog(self.entries + tuple(entries))

    def of_kind(self, kind: str) -> list[KnowledgeEntry]:
        return [entry for entry in self.entries if entry.kind == kind]

    def __iter__(self) -> Iterator[KnowledgeEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_list(self) -> list[dict[str, Any]]:
        """JSON-safe list of entries (newest last)."""
        return [entry.to_dict() for entry in self.entries]
