"""Week 3 lifecycle contract tests. Implemented for user/reviewer execution."""
from decimal import Decimal
import unittest
from unittest.mock import patch

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState
from dumbocracy.errors import DumbocracyError, ErrorCode


def fixture_config(horizon: int = 2, budget: str = "10.00") -> GameConfig:
    return GameConfig(SeedConfig(17), GraphSpec("edge_list", 3), horizon, Decimal(budget), DynamicsConfig(.5), ActionConfig(.2, .1))


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"), ("v1", "v2")))
        self.voters = (VoterState("v0", .2, .5), VoterState("v1", .5, .5), VoterState("v2", .8, .5))
        self.engine = GameEngine()

    def test_targeted_action_lifecycle_and_terminal_result(self) -> None:
        state = self.engine.initialize(fixture_config(), self.graph, self.voters)
        response = self.engine.step(state, CampaignAction("targeted_contact", ("v0",)))
        self.assertTrue(response.accepted)
        self.assertEqual([v.support for v in response.state.voters], [.375, .5, .7250000000000001])
        self.assertEqual(response.state.ledger.remaining_budget, Decimal("9.00"))
        terminal = self.engine.step(response.state, CampaignAction("no_op"))
        self.assertTrue(terminal.terminal)
        self.assertEqual(terminal.state.round_index, 2)
        self.assertEqual(terminal.state.result.vote_share, 2 / 3)
        self.assertTrue(terminal.state.result.victory)
        payload = terminal.to_public_dict()
        self.assertTrue(payload["observation"]["terminal_status"])
        self.assertEqual(payload["result"]["vote_share"], 2 / 3)
        self.assertEqual(payload["observation"]["legal_actions"], [])

    def test_rejection_keeps_round_and_money_and_terminal_is_guarded(self) -> None:
        state = self.engine.initialize(fixture_config(1), self.graph, self.voters)
        rejected = self.engine.step(state, CampaignAction("targeted_contact", ("unknown",)))
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.state.round_index, 0)
        self.assertEqual(rejected.state.ledger, state.ledger)
        self.assertTrue(rejected.state.trace[-1].action_atomic)
        self.assertEqual(rejected.state.trace[-1].state_fingerprint_before, rejected.state.trace[-1].state_fingerprint_after)
        self.assertTrue(rejected.state.trace[-1].predicates()["ActionAtomic"])
        finished = self.engine.step(state, CampaignAction("no_op"))
        after = self.engine.step(finished.state, CampaignAction("no_op"))
        self.assertFalse(after.accepted)
        self.assertEqual(after.error_code, "TERMINAL_STATE")
        self.assertEqual(after.state.result, finished.state.result)
        self.assertNotIn("action_accepted", [event.event_type for event in rejected.state.internal_events])

    def test_insufficient_budget_does_not_change_authoritative_fields(self) -> None:
        state = self.engine.initialize(fixture_config(2, "0.50"), self.graph, self.voters)
        rejected = self.engine.step(state, CampaignAction("targeted_contact", ("v0",)))
        self.assertEqual(rejected.error_code, "INSUFFICIENT_BUDGET")
        self.assertEqual(rejected.state.voters, state.voters)
        self.assertEqual(rejected.state.ledger, state.ledger)
        self.assertEqual(rejected.state.round_index, state.round_index)
        public_error = rejected.to_public_dict()
        self.assertEqual(public_error["error_code"], "INSUFFICIENT_BUDGET")
        self.assertIsNone(public_error["result"])
        self.assertNotIn("root_seed", str(public_error))

    def test_budget_exhaustion_does_not_terminate_and_effect_expires(self) -> None:
        state = self.engine.initialize(fixture_config(2, "1.00"), self.graph, self.voters)
        spent = self.engine.step(state, CampaignAction("targeted_contact", ("v0",)))
        self.assertTrue(spent.accepted)
        self.assertFalse(spent.terminal)
        self.assertEqual(spent.state.ledger.remaining_budget, Decimal("0.00"))
        self.assertEqual(spent.state.current_contributions, (0.0, 0.0, 0.0))
        self.assertEqual([item.action_id for item in spent.observation.legal_actions], ["no_op"])
        finished = self.engine.step(spent.state, CampaignAction("no_op"))
        self.assertTrue(finished.terminal)

    def test_numerical_failure_rolls_back_resource_commit(self) -> None:
        state = self.engine.initialize(fixture_config(), self.graph, self.voters)
        with patch("dumbocracy.engine.advance_support", side_effect=DumbocracyError(ErrorCode.NUMERICAL_INSTABILITY, "safe numeric failure")):
            failed = self.engine.step(state, CampaignAction("targeted_contact", ("v0",)))
        self.assertFalse(failed.accepted)
        self.assertEqual(failed.error_code, "NUMERICAL_INSTABILITY")
        self.assertEqual(failed.state.voters, state.voters)
        self.assertEqual(failed.state.ledger, state.ledger)
        self.assertEqual(failed.state.round_index, state.round_index)

    def test_internal_event_order_is_stable(self) -> None:
        state = self.engine.initialize(fixture_config(1), self.graph, self.voters)
        completed = self.engine.step(state, CampaignAction("targeted_contact", ("v0",))).state
        self.assertEqual(
            [event.event_type for event in completed.internal_events[-6:]],
            ["action_accepted", "resource_spent", "voter_update_summary", "round_completed", "game_terminated", "observation_generated"],
        )

    def test_broadcast_guard_resets_at_round_boundary(self) -> None:
        state = self.engine.initialize(fixture_config(), self.graph, self.voters)
        first = self.engine.step(state, CampaignAction("broadcast"))
        self.assertTrue(first.accepted)
        self.assertFalse(first.state.broadcast_used)
        second = self.engine.step(first.state, CampaignAction("broadcast"))
        self.assertTrue(second.accepted)
        self.assertEqual(second.state.ledger.spent, Decimal("10.00"))

    def test_reset_and_replay_determinism(self) -> None:
        initial = self.engine.initialize(fixture_config(), self.graph)
        reset = self.engine.reset(self.engine.step(initial, CampaignAction("no_op")).state)
        self.assertEqual(self.engine.observe(initial).to_public_dict(), self.engine.observe(reset).to_public_dict())
        actions = (CampaignAction("targeted_contact", ("v0",)), CampaignAction("no_op"))
        left = self.engine.replay(fixture_config(), actions, self.graph)
        right = self.engine.replay(fixture_config(), actions, self.graph)
        self.assertEqual([self.engine.observe(s).to_public_dict() for s in left], [self.engine.observe(s).to_public_dict() for s in right])
        self.assertEqual(left[-1].result, right[-1].result)
        rejected_replay_a = self.engine.replay(fixture_config(), (CampaignAction("targeted_contact", ("missing",)), CampaignAction("no_op")), self.graph)
        rejected_replay_b = self.engine.replay(fixture_config(), (CampaignAction("targeted_contact", ("missing",)), CampaignAction("no_op")), self.graph)
        self.assertEqual([s.public_events for s in rejected_replay_a], [s.public_events for s in rejected_replay_b])
        self.assertEqual(rejected_replay_a[-1].round_index, 1)

    def test_exact_half_vote_is_not_counted(self) -> None:
        graph = ElectorateGraph(("v0", "v1", "v2", "v3"), ())
        voters = (VoterState("v0", .5, 0), VoterState("v1", .5, 0), VoterState("v2", .8, 0), VoterState("v3", .7, 0))
        config = GameConfig(SeedConfig(17), GraphSpec("edge_list", 4), 1, Decimal("10.00"), DynamicsConfig(.5), ActionConfig(.2, .1))
        terminal = self.engine.step(self.engine.initialize(config, graph, voters), CampaignAction("no_op"))
        self.assertEqual(terminal.state.result.player_votes, 2)
        self.assertEqual(terminal.state.result.vote_share, .5)
        self.assertFalse(terminal.state.result.victory)

