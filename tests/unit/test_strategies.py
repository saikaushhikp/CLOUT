"""Observation-only strategy contracts; authored, not executed here."""
import random
import unittest

from dumbocracy.information import ActionDescriptor, Observation
from dumbocracy.strategies import (
    FirstLegalStrategy, HistoryEntry, NoOpStrategy, PublicActionSummary,
    SeededRandomLegalStrategy, StrategyError,
    UniformBudgetStrategy, GreedyPublicModelStrategy,
)
from dumbocracy.graph import GraphSummary


def descriptor(action_id, targets=()):
    return ActionDescriptor(action_id, "0.00", "public", (), tuple(targets))


def observation(legal):
    return Observation(0, "10.00", .5, GraphSummary(2, 1, 1.0, 1, 1, 1.0, 1), tuple(legal), False, ())


class StrategyTests(unittest.TestCase):
    def test_baselines_return_only_supplied_supported_campaign_actions(self):
        legal = (descriptor("no_op"), descriptor("broadcast"), descriptor("targeted_contact", ("v1", "v0")))
        obs = observation(legal)
        for policy in (NoOpStrategy(), FirstLegalStrategy(), SeededRandomLegalStrategy()):
            action = policy.choose(obs, (), legal, random.Random(10))
            self.assertIn(action.action_id, {item.action_id for item in legal})
            self.assertIsInstance(action.target_ids, tuple)
            if action.action_id == "targeted_contact":
                self.assertEqual(len(action.target_ids), 1)
                self.assertIn(action.target_ids[0], {"v0", "v1"})

    def test_stable_first_legal_order_and_target_selection(self):
        legal = (descriptor("no_op"), descriptor("targeted_contact", ("v2", "v1")), descriptor("broadcast"))
        result = FirstLegalStrategy().choose(observation(legal), (), legal, random.Random(1))
        self.assertEqual(result.action_id, "targeted_contact")
        self.assertEqual(result.target_ids, ("v1",))

    def test_first_legal_falls_back_when_targeted_or_broadcast_are_unavailable(self):
        broadcast_only = (descriptor("broadcast"), descriptor("no_op"))
        self.assertEqual(FirstLegalStrategy().choose(observation(broadcast_only), (), broadcast_only, random.Random(1)).action_id, "broadcast")
        noop_only = (descriptor("no_op"),)
        self.assertEqual(FirstLegalStrategy().choose(observation(noop_only), (), noop_only, random.Random(1)).action_id, "no_op")

    def test_seeded_random_sequence_repeats(self):
        legal = (descriptor("no_op"), descriptor("broadcast"), descriptor("targeted_contact", ("v0", "v1", "v2")))
        policy = SeededRandomLegalStrategy()
        left_rng, right_rng = random.Random(55), random.Random(55)
        left = [policy.choose(observation(legal), (), legal, left_rng) for _ in range(20)]
        right = [policy.choose(observation(legal), (), legal, right_rng) for _ in range(20)]
        self.assertEqual(left, right)

    def test_noop_strategy_requires_legal_noop(self):
        with self.assertRaises(StrategyError):
            NoOpStrategy().choose(observation((descriptor("broadcast"),)), (), (descriptor("broadcast"),), random.Random(1))

    def test_targeted_descriptor_without_public_targets_is_rejected(self):
        legal = (descriptor("targeted_contact"),)
        with self.assertRaises(StrategyError):
            FirstLegalStrategy().choose(observation(legal), (), legal, random.Random(1))

    def test_empty_legal_list_fails_clearly(self):
        with self.assertRaises(StrategyError):
            FirstLegalStrategy().choose(observation(()), (), (), random.Random(1))

    def test_uniform_and_greedy_are_deterministic_legal_public_policies(self):
        legal = (ActionDescriptor("targeted_contact", "1.00 per target", "one", (("delta_target", .2),), ("v1", "v0")),
                 ActionDescriptor("broadcast", "5.00", "all", (("delta_broadcast", .1),)), descriptor("no_op"))
        obs = observation(legal)
        greedy = GreedyPublicModelStrategy(model_seed=7)
        self.assertEqual(greedy.choose(obs, (), legal, random.Random(3)), greedy.choose(obs, (), legal, random.Random(3)))
        for policy in (UniformBudgetStrategy(5), greedy):
            result = policy.choose(obs, (), legal, random.Random(3))
            self.assertIn(result.action_id, {item.action_id for item in legal})
            if result.action_id == "targeted_contact":
                self.assertIn(result.target_ids[0], ("v0", "v1"))

    def test_history_is_prior_public_data_and_has_no_engine_state_field(self):
        obs = observation((descriptor("no_op"),))
        prior = HistoryEntry(obs, PublicActionSummary(0, "no_op", True, None))
        self.assertEqual(prior.observation.round_index, 0)
        self.assertFalse(hasattr(prior, "state"))
        self.assertFalse(hasattr(FirstLegalStrategy(), "engine"))


if __name__ == "__main__":
    unittest.main()
