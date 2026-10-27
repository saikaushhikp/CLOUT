"""Observation-only baseline strategies for reproducible episode runs."""
from __future__ import annotations

from dataclasses import dataclass
import random
from decimal import Decimal
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


@dataclass(frozen=True, slots=True)
class UniformBudgetStrategy:
    """Spend at a steady pace, reserving enough for later rounds where possible."""

    horizon_rounds: int = 5
    name: str = "uniform_budget"
    version: str = "1"

    def choose(self, observation: Observation, history: tuple[HistoryEntry, ...], legal_actions: tuple[ActionDescriptor, ...], rng: random.Random) -> CampaignAction:
        del history
        descriptors = _supported_descriptors(legal_actions)
        remaining_rounds = max(1, self.horizon_rounds - observation.round_index)
        budget = Decimal(observation.remaining_budget)
        affordable = [item for item in descriptors if Decimal(item.cost.split()[0]) <= budget]
        target = next((item for item in affordable if item.action_id == "targeted_contact"), None)
        broadcast = next((item for item in affordable if item.action_id == "broadcast"), None)
        if broadcast and budget >= Decimal(broadcast.cost) * remaining_rounds:
            return _proposal(broadcast)
        if target and budget >= Decimal(target.cost.split()[0]) * remaining_rounds:
            return _proposal(target, rng)
        no_op = next((item for item in descriptors if item.action_id == "no_op"), None)
        if budget == 0 and no_op:
            return _proposal(no_op)
        candidates = [item for item in (broadcast, target, no_op) if item]
        if not candidates:
            raise StrategyError("no affordable supported action was supplied")
        return _proposal(candidates[0], rng)


@dataclass(frozen=True, slots=True)
class GreedyPublicModelStrategy:
    """Small aggregate-only scenario-prior policy; intentionally not an optimizer proof."""

    model_seed: int = 0
    name: str = "greedy_public_model"
    version: str = "1"

    def choose(self, observation: Observation, history: tuple[HistoryEntry, ...], legal_actions: tuple[ActionDescriptor, ...], rng: random.Random) -> CampaignAction:
        del history
        descriptors = _supported_descriptors(legal_actions)
        budget = Decimal(observation.remaining_budget)
        samples = random.Random(self.model_seed ^ (observation.round_index * 0x9E3779B1))
        candidates: list[tuple[float, tuple[int, str], CampaignAction]] = []
        for descriptor in descriptors:
            if descriptor.action_id == "targeted_contact":
                cost = Decimal(descriptor.cost.split()[0])
                if cost > budget or not descriptor.eligible_target_ids:
                    continue
                action = _proposal(descriptor, rng)
                delta = dict(descriptor.parameters).get("delta_target", 0.0)
            elif descriptor.action_id == "broadcast":
                cost = Decimal(descriptor.cost)
                if cost > budget:
                    continue
                action = _proposal(descriptor)
                delta = dict(descriptor.parameters).get("delta_broadcast", 0.0)
            else:
                action = _proposal(descriptor)
                delta = 0.0
                cost = Decimal("0.00")
            # Declared synthetic exchangeable prior: 64 scenarios, 256 voters,
            # Beta concentration 4 around the public aggregate; no real graph or
            # voter features. One action affects all (broadcast) or a uniformly
            # random singleton (targeted), with neutral mean-field drift.
            n = min(256, observation.graph_summary.node_count)
            mean = min(.999, max(.001, observation.aggregate_support))
            shape_a, shape_b = mean * 4, (1 - mean) * 4
            wins = 0
            for _ in range(64):
                supports = [samples.betavariate(shape_a, shape_b) for _ in range(n)]
                if descriptor.action_id == "broadcast":
                    supports = [min(1.0, value + delta) for value in supports]
                elif descriptor.action_id == "targeted_contact" and supports:
                    target_index = samples.randrange(n)
                    supports[target_index] = min(1.0, supports[target_index] + delta)
                wins += sum(value > .5 for value in supports) / n
            estimated = wins / 64 - (float(cost / max(budget, Decimal("1.00"))) * 1e-4)
            candidates.append((estimated, (-int(cost * 100), descriptor.action_id), action))
        if not candidates:
            raise StrategyError("no supported affordable action was supplied")
        return max(candidates, key=lambda row: (row[0], row[1]))[2]
