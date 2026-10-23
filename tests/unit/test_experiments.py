"""Paired-seed experiment contracts; authored, not executed here."""
from decimal import Decimal
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.experiments import ExperimentScenario, aggregate_records, run_paired_experiments, run_paired_scenarios
from dumbocracy.engine import GameEngine
from dumbocracy.graph import ElectorateGraph
from dumbocracy.strategies import FirstLegalStrategy, NoOpStrategy


def config():
    return GameConfig(SeedConfig(1), GraphSpec("edge_list", 3), 2, Decimal("10.00"), DynamicsConfig(.3), ActionConfig(.1, .05))


def factories():
    return {"noop": NoOpStrategy, "first": FirstLegalStrategy}


class ExperimentTests(unittest.TestCase):
    def test_paired_seed_records_are_complete_and_jsonl_round_trips(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"),))
        report = run_paired_experiments(config(), (17,), factories(), graph=graph)
        self.assertEqual(len(report.records), 2)
        self.assertEqual({row["paired_run_id"] for row in report.records}, {"default:17"})
        self.assertEqual([row["root_seed"] for row in report.records], [17, 17])
        for line in report.to_jsonl().splitlines():
            self.assertIsInstance(json.loads(line), dict)
        self.assertEqual({row["status"] for row in report.records}, {"completed"})

    def test_multiple_scenarios_are_paired_for_each_seed_and_policy(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"),))
        er_config = replace(config(), graph=GraphSpec("erdos_renyi", 4, .5))
        scenarios = (ExperimentScenario("fixture", config(), graph), ExperimentScenario("er4", er_config))
        report = run_paired_scenarios(scenarios, (17,), factories())
        self.assertEqual(len(report.records), 4)
        self.assertEqual({row["scenario_id"] for row in report.records}, {"fixture", "er4"})
        for scenario_id in ("fixture", "er4"):
            paired = [row for row in report.records if row["scenario_id"] == scenario_id]
            self.assertEqual(len(paired), 2)
            self.assertEqual({row["paired_run_id"] for row in paired}, {f"{scenario_id}:17"})

    def test_identical_run_repeats_actions_results_and_stable_record_fields(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"),))
        left = run_paired_experiments(config(), (17,), {"noop": NoOpStrategy}, graph=graph).records[0]
        right = run_paired_experiments(config(), (17,), {"noop": NoOpStrategy}, graph=graph).records[0]
        for key in ("action_sequence", "terminal_result", "logic_diagnostics", "root_seed", "derived_seeds", "graph", "game", "runtime"):
            self.assertEqual(left[key], right[key])

    def test_policies_use_deterministic_same_initialization_per_seed(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"),))
        original_initialize = GameEngine.initialize
        initial_snapshots = []

        def capture_initialize(engine, game_config, graph=None, voters=None):
            state = original_initialize(engine, game_config, graph=graph, voters=voters)
            initial_snapshots.append((state.graph.edges, state.voters))
            return state

        with patch.object(GameEngine, "initialize", capture_initialize):
            report = run_paired_experiments(config(), (17,), factories(), graph=graph)
        self.assertEqual(initial_snapshots[0], initial_snapshots[1])
        self.assertEqual(report.records[0]["graph"], report.records[1]["graph"])
        # The runner uses the same immutable config/graph and subsystem seeds,
        # then initializes a fresh independent engine state for each policy.
        self.assertEqual(report.records[0]["derived_seeds"], report.records[1]["derived_seeds"])

    def test_rejected_policy_action_is_one_failure_not_an_infinite_retry(self):
        class InvalidPolicy:
            name = "invalid"
            version = "1"

            def choose(self, observation, history, legal_actions, rng):
                from dumbocracy.campaign import CampaignAction
                return CampaignAction("targeted_contact", ("not-publicly-eligible",))

        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), ())
        record = run_paired_experiments(config(), (2,), {"invalid": InvalidPolicy}, graph=graph).records[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["rejected_count"], 1)
        self.assertEqual(len(record["action_sequence"]), 1)
        self.assertEqual(record["error_code"], "INVALID_TARGET")
        self.assertIsNone(record["action_sequence"][0]["target_ids"])

    def test_only_publicly_eligible_targets_are_serialized(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), ())
        record = run_paired_experiments(config(), (2,), {"first": FirstLegalStrategy}, graph=graph).records[0]
        for proposal in record["action_sequence"]:
            if proposal["action_id"] == "targeted_contact":
                self.assertEqual(len(proposal["target_ids"]), 1)
                self.assertIn(proposal["target_ids"][0], {"v0", "v1", "v2"})

    def test_aggregate_metrics_have_defined_denominators(self):
        records = (
            {"strategy": "x", "strategy_version": "1", "status": "completed", "terminal_result": {"score": .25, "victory": False, "budget_used": "2.00"}, "rejected_count": 0, "action_sequence": [{"accepted": True}]},
            {"strategy": "x", "strategy_version": "1", "status": "completed", "terminal_result": {"score": .75, "victory": True, "budget_used": "4.00"}, "rejected_count": 1, "action_sequence": [{"accepted": False}]},
            {"strategy": "x", "strategy_version": "1", "status": "failed", "rejected_count": 1, "action_sequence": [{"accepted": False}]},
        )
        aggregate = aggregate_records(records)[0]
        self.assertEqual(aggregate["sample_count"], 3)
        self.assertEqual(aggregate["completed_count"], 2)
        self.assertEqual(aggregate["failed_episode_count"], 1)
        self.assertEqual(aggregate["mean_terminal_score"], .5)
        self.assertEqual(aggregate["terminal_score_population_stddev"], .25)
        self.assertEqual(aggregate["victory_rate_completed"], .5)
        self.assertEqual(aggregate["mean_budget_used_completed"], 3.0)
        self.assertEqual(aggregate["rejected_proposal_rate"], 2 / 3)

    def test_output_does_not_include_hidden_voter_data(self):
        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"),))
        report = run_paired_experiments(config(), (4,), {"noop": NoOpStrategy}, graph=graph)
        rendered = report.to_jsonl().lower()
        for forbidden in ("susceptibility", "voter_state", "support_values", "initial_voters", "device_graph"):
            self.assertNotIn(forbidden, rendered)

    def test_empty_or_duplicate_seed_sets_rejected(self):
        with self.assertRaises(ValueError):
            run_paired_experiments(config(), (), factories())
        with self.assertRaises(ValueError):
            run_paired_experiments(config(), (1, 1), factories())

    def test_strategy_exception_is_counted_as_a_failed_episode(self):
        class FailingPolicy:
            name = "failure"
            version = "1"

            def choose(self, observation, history, legal_actions, rng):
                raise RuntimeError("detail must not be serialized")

        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), ())
        report = run_paired_experiments(config(), (8,), {"failure": FailingPolicy}, graph=graph)
        self.assertEqual(report.aggregates[0]["failed_episode_count"], 1)
        self.assertEqual(report.records[0]["failure_type"], "RuntimeError")
        self.assertNotIn("detail must not be serialized", report.to_jsonl())

    def test_runner_selects_once_per_active_round_and_stops_at_terminal(self):
        calls = []

        class CountingNoOp:
            name = "counting_noop"
            version = "1"

            def choose(self, observation, history, legal_actions, rng):
                from dumbocracy.campaign import CampaignAction
                calls.append(observation.round_index)
                return CampaignAction("no_op")

        graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), ())
        run_paired_experiments(config(), (9,), {"count": CountingNoOp}, graph=graph)
        self.assertEqual(calls, [0, 1])


if __name__ == "__main__":
    unittest.main()
