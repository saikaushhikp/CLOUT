"""Player protocol (spec §8): a player maps its allowlisted observation to an action dict.

Action dicts are the JSON form accepted by :func:`dumbocracy.game.action_from_dict`::

    {"type": "query", "sector_a": "Edu", "sector_b": "Agri", "region": "N"}
    {"type": "commit", "manifesto": [0.2, 0.2, 0.2, 0.2, 0.2]}
    {"type": "allocate", "hires_by_region": [2, 0, 1, 0], "media_regions": [False, True, False, False]}
    {"type": "forfeit"}
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Player(Protocol):
    """Anything that can play: the human (via the HTTP API), scripted strategies, an LLM (future)."""

    player_id: str

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return the next action dict given the contract ``Observation`` (web_api_v2.md)."""
        ...
