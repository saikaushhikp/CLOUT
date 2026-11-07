"""Dumbocracy v2: an election-campaign game over a GPU-simulated multi-agent electorate.

Source of truth: ``docs/v2/technical_spec_v2.md``. Phase plans: ``docs/v2/plans/``.
"""
from .config import (
    ConfigError, DynamicsConfig, ElectorateConfig, GameConfig, MediaConfig, SeedStreams,
    SurveyConfig, VotingConfig, WorkerConfig, resolve_device,
)
from .sectors import ARCHETYPES, PERSONALITIES, REGIONS, SECTORS

__version__ = "2.0.0.dev0"
__all__ = [
    "ConfigError", "DynamicsConfig", "ElectorateConfig", "GameConfig", "MediaConfig", "SeedStreams",
    "SurveyConfig", "VotingConfig", "WorkerConfig", "resolve_device",
    "ARCHETYPES", "PERSONALITIES", "REGIONS", "SECTORS",
]
