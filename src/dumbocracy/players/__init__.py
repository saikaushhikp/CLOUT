"""Player protocol and implementations: human/API, scripted, LLM stub (P3, spec §8)."""
from .base import Player
from .llm_stub import LLMPlayer

__all__ = ["LLMPlayer", "Player"]
