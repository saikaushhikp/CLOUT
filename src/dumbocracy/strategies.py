"""Observation-only baseline strategies for reproducible episode runs."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol, Sequence

from .campaign import CampaignAction
from .information import ActionDescriptor, Observation


class StrategyError(ValueError):
    """A strategy cannot form a proposal from the supplied public options."""


@dataclass(frozen=True, slots=True)
class PublicActionSummary:
    round_index: int
    action_id: str
    accepted: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One prior decision boundary; current observation is not in history."""

    observation: Observation
    action_result: PublicActionSummary


class Strategy(Protocol):
    name: str
    version: str

    def choose(
        self,
        observation: Observation,
        history: tuple[HistoryEntry, ...],
        legal_actions: tuple[ActionDescriptor, ...],
        rng: random.Random,
    ) -> CampaignAction: ...


def _supported_descriptors(legal_actions: Sequence[ActionDescriptor]) -> tuple[ActionDescriptor, ...]:
    if any(not isinstance(descriptor, ActionDescriptor) for descriptor in legal_actions):
        raise StrategyError("legal_actions must contain public ActionDescriptor values")
    supported = {"targeted_contact", "broadcast", "no_op"}
    return tuple(sorted(
        (descriptor for descriptor in legal_actions if descriptor.action_id in supported),
        key=lambda descriptor: ({"targeted_contact": 0, "broadcast": 1, "no_op": 2}[descriptor.action_id], descriptor.action_id),
    ))


def _proposal(descriptor: ActionDescriptor, rng: random.Random | None = None) -> CampaignAction:
    if descriptor.action_id == "targeted_contact":
        if any(not isinstance(target, str) for target in descriptor.eligible_target_ids):
            raise StrategyError("eligible target identifiers must be public strings")
        eligible = tuple(sorted(set(descriptor.eligible_target_ids)))
        if not eligible:
            raise StrategyError("targeted_contact is listed without an eligible public target")
        target = eligible[0] if rng is None else rng.choice(eligible)
        return CampaignAction("targeted_contact", (target,))
    if descriptor.action_id == "broadcast":
        return CampaignAction("broadcast")
    if descriptor.action_id == "no_op":
        return CampaignAction("no_op")
    raise StrategyError("unsupported legal-action descriptor")


@dataclass(frozen=True, slots=True)
class NoOpStrategy:
    name: str = "no_op"
    version: str = "1"

    def choose(self, observation: Observation, history: tuple[HistoryEntry, ...], legal_actions: tuple[ActionDescriptor, ...], rng: random.Random) -> CampaignAction:
        del observation, history, rng
        descriptor = next((item for item in _supported_descriptors(legal_actions) if item.action_id == "no_op"), None)
        if descriptor is None:
            raise StrategyError("no_op is not among supplied legal actions")
        return _proposal(descriptor)


@dataclass(frozen=True, slots=True)
class FirstLegalStrategy:
    """Choose by fixed priority targeted_contact, broadcast, then no_op."""

    name: str = "first_legal"
    version: str = "1"

    def choose(self, observation: Observation, history: tuple[HistoryEntry, ...], legal_actions: tuple[ActionDescriptor, ...], rng: random.Random) -> CampaignAction:
        del observation, history, rng
        descriptors = _supported_descriptors(legal_actions)
        if not descriptors:
            raise StrategyError("no supported legal action was supplied")
        return _proposal(descriptors[0])


@dataclass(frozen=True, slots=True)
class SeededRandomLegalStrategy:
    name: str = "seeded_random_legal"
    version: str = "1"

    def choose(self, observation: Observation, history: tuple[HistoryEntry, ...], legal_actions: tuple[ActionDescriptor, ...], rng: random.Random) -> CampaignAction:
        del observation, history
        descriptors = _supported_descriptors(legal_actions)
        if not descriptors:
            raise StrategyError("no supported legal action was supplied")
        return _proposal(rng.choice(descriptors), rng)
