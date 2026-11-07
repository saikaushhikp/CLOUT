"""LiteLLM-shaped opponent stub (spec §8, decision U3: interface only, not implemented).

Prompt / observation contract for a future implementation
---------------------------------------------------------
* Input: exactly the allowlisted ``Observation`` dict of ``docs/v2/web_api_v2.md`` for this player
  (never the engine state). It is serialised to JSON and placed in the user message, together with
  ``observation["legal_actions"]`` and the public rules (costs are already inside ``costs``).
* System prompt: the game rules of spec §3–§7 in prose, the sector/region ids, and the instruction
  to answer with ONE JSON object matching an action dict (see :mod:`.base`), nothing else.
* Output: parsed with ``json.loads`` and validated by the engine; a rejected action (``accepted:
  false`` with an ``error_code``) is fed back as a follow-up message for at most ``max_retries``
  attempts, after which the player sends ``{"type": "forfeit"}`` in ROUND (or a uniform commit in
  SURVEY).
* Calls: ``litellm.completion(model=self.model, messages=..., temperature=self.temperature,
  response_format={"type": "json_object"})``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMPlayer:
    """Future LLM opponent. Every method raises ``NotImplementedError`` (spec §13: out of scope)."""

    player_id: str = "p1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_retries: int = 2

    def system_prompt(self) -> str:
        raise NotImplementedError("LLMPlayer is an interface stub (spec §8, U3)")

    def build_messages(self, observation: dict[str, Any]) -> list[dict[str, str]]:
        raise NotImplementedError("LLMPlayer is an interface stub (spec §8, U3)")

    def parse_action(self, completion_text: str) -> dict[str, Any]:
        raise NotImplementedError("LLMPlayer is an interface stub (spec §8, U3)")

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("LLMPlayer is an interface stub (spec §8, U3)")
