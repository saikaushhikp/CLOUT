"""Hand-computable bounded temporal diagnostics; authored, not executed here."""
import json
from pathlib import Path
import unittest
from decimal import Decimal

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState
from dumbocracy.logic import (
    ATOMS, And, Atom, CanonicalTrace, Eventually, Globally, Implies,
    LogicInputError, LogicStatus, Not, Or, TraceState, evaluate_diagnostics, evaluate_trace,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "logic"


def load_fixture(name: str) -> CanonicalTrace:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return CanonicalTrace.from_dict(data)


class LogicTests(unittest.TestCase):
    def test_three_approved_diagnostics_on_valid_trace(self):
        diagnostics = evaluate_diagnostics(load_fixture("valid_terminal_at_horizon.json"))
        self.assertEqual([result.status for result in diagnostics.values()], [LogicStatus.SATISFIED] * 3)
        self.assertEqual(diagnostics["F_<=H Terminal (H=2)"].witness_index, 2)

    def test_safety_reports_first_counterexample(self):
        result = evaluate_trace(
            load_fixture("safety_false_at_one.json"),
            Globally(And(Atom("BudgetNonnegative"), Atom("StateValid"))),
        )
        self.assertEqual(result.status, LogicStatus.VIOLATED)
        self.assertEqual(result.counterexample_index, 1)

    def test_missing_terminal_is_false_not_invalid(self):
        trace = load_fixture("missing_terminal_at_bound.json")
        result = evaluate_trace(trace, Eventually(Atom("Terminal"), trace.horizon))
        self.assertEqual(result.status, LogicStatus.VIOLATED)

    def test_one_step_bound_excludes_terminal_beyond_bound(self):
        trace = load_fixture("valid_terminal_at_horizon.json")
        result = evaluate_trace(trace, Eventually(Atom("Terminal"), 1))
        self.assertEqual(result.status, LogicStatus.VIOLATED)
        self.assertEqual(result.checked_interval, (0, 1))

    def test_rejection_audit_does_not_create_game_time(self):
        trace = load_fixture("rejection_same_time_index.json")
        self.assertEqual(len(trace.states), trace.horizon + 1)
        self.assertEqual(trace.states[0].predicate("ActionAtomic"), True)
        self.assertEqual({event.round_index for event in trace.audit_events}, {0})
        predicates = trace.states[0].to_public_dict()["predicates"]
        predicates["ActionAtomic"] = False
        falsified = CanonicalTrace(1, (TraceState.from_mapping(0, predicates), trace.states[1]), trace.audit_events)
        self.assertEqual(evaluate_trace(falsified, Atom("ActionAtomic")).status, LogicStatus.VIOLATED)

    def test_completed_engine_trace_adapter_folds_rejection_by_round(self):
        config = GameConfig(SeedConfig(7), GraphSpec("edge_list", 2), 2, Decimal("5.00"), DynamicsConfig(.2), ActionConfig(.1, .1))
        graph = ElectorateGraph.from_edge_ids(("v0", "v1"), (("v0", "v1"),))
        snapshots = GameEngine().replay(config, (CampaignAction("targeted_contact", ("bad",)), CampaignAction("no_op"), CampaignAction("no_op")), graph)
        trace = CanonicalTrace.from_game_state(snapshots[-1])
        self.assertEqual(len(trace.states), 3)
        self.assertEqual(trace.states[0].predicate("ActionAtomic"), True)
        self.assertTrue(any(event.event_type == "action_rejected" and event.round_index == 0 for event in trace.audit_events))

    def test_unknowns_propagate_with_kleene_boolean_rules(self):
        trace = load_fixture("unknown_atoms.json")
        self.assertEqual(evaluate_trace(trace, Atom("BudgetNonnegative")).status, LogicStatus.UNKNOWN)
        self.assertEqual(evaluate_trace(trace, And(Atom("BudgetNonnegative"), Atom("StateValid"))).status, LogicStatus.UNKNOWN)
        self.assertEqual(evaluate_trace(trace, Or(Atom("BudgetNonnegative"), Atom("Terminal"))).status, LogicStatus.UNKNOWN)
        self.assertEqual(evaluate_trace(trace, And(Atom("BudgetNonnegative"), Not(Atom("StateValid")))).status, LogicStatus.VIOLATED)
        self.assertEqual(evaluate_trace(trace, Or(Atom("BudgetNonnegative"), Atom("StateValid"))).status, LogicStatus.SATISFIED)
        self.assertEqual(evaluate_trace(trace, Not(Atom("BudgetNonnegative"))).status, LogicStatus.UNKNOWN)
        self.assertEqual(evaluate_trace(trace, Implies(Atom("Terminal"), Atom("BudgetNonnegative"))).status, LogicStatus.SATISFIED)
        self.assertEqual(evaluate_trace(trace, Globally(Atom("BudgetNonnegative"))).status, LogicStatus.UNKNOWN)
        self.assertEqual(evaluate_trace(trace, Eventually(Atom("BudgetNonnegative"), 0)).status, LogicStatus.UNKNOWN)

    def test_zero_bound_and_inclusive_endpoint(self):
        trace = load_fixture("exact_half_is_defeat.json")
        self.assertEqual(evaluate_trace(trace, Eventually(Atom("Terminal"), 0)).witness_index, 0)
        self.assertEqual(evaluate_trace(trace, Globally(Atom("Victory"), 0)).status, LogicStatus.VIOLATED)

    def test_exact_half_outcome_is_false_victory(self):
        trace = load_fixture("exact_half_is_defeat.json")
        self.assertFalse(trace.states[0].predicate("Victory"))
        self.assertEqual(evaluate_trace(trace, Atom("Victory")).status, LogicStatus.VIOLATED)

    def test_engine_exact_half_result_maps_to_false_victory_atom(self):
        config = GameConfig(SeedConfig(3), GraphSpec("edge_list", 4), 1, Decimal("1.00"), DynamicsConfig(0), ActionConfig(0, 0))
        graph = ElectorateGraph(("v0", "v1", "v2", "v3"), ())
        voters = (VoterState("v0", .5, 0), VoterState("v1", .5, 0), VoterState("v2", .8, 0), VoterState("v3", .7, 0))
        final = GameEngine().step(GameEngine().initialize(config, graph, voters), CampaignAction("no_op")).state
        self.assertEqual(final.result.vote_share, .5)
        self.assertFalse(final.result.victory)
        trace = CanonicalTrace.from_game_state(final)
        self.assertFalse(trace.states[-1].predicate("Victory"))

    def test_invalid_atom_and_bound_rejected(self):
        trace = load_fixture("valid_terminal_at_horizon.json")
        with self.assertRaises(LogicInputError):
            evaluate_trace(trace, Atom("SecretSupport"))
        with self.assertRaises(LogicInputError):
            evaluate_trace(trace, Eventually(Atom("Terminal"), -1))

    def test_malformed_trace_rejected_separately_from_formula_result(self):
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        del source["states"][0]["StateValid"]
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][0]["StateValid"] = 1
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"].pop()
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)

    def test_terminal_before_end_and_duplicate_terminal_are_invalid(self):
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][0]["Terminal"] = True
        source["states"][0]["Victory"] = True
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)

    def test_invalid_preterminal_atoms_and_noncontiguous_time_rejected(self):
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][0]["Victory"] = False
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        # Explicit state-object form includes predicate map and time index.
        source["states"][1] = {"time_index": 8, "predicates": source["states"][1]}
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["audit_events"] = [
            {"sequence": 4, "round_index": 0, "event_type": "action_rejected"},
            {"sequence": 3, "round_index": 0, "event_type": "action_proposed"},
        ]
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][0]["SecretSupport"] = True
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][1]["Terminal"] = True
        source["states"][1]["Victory"] = True
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)
        source = json.loads((FIXTURES / "valid_terminal_at_horizon.json").read_text())
        source["states"][-1]["Victory"] = None
        with self.assertRaises(LogicInputError):
            CanonicalTrace.from_dict(source)

    def test_evaluation_json_safe_and_deterministic(self):
        trace = load_fixture("valid_terminal_at_horizon.json")
        left = evaluate_diagnostics(trace)
        right = evaluate_diagnostics(trace)
        left_payload = {name: result.to_public_dict() for name, result in left.items()}
        right_payload = {name: result.to_public_dict() for name, result in right.items()}
        self.assertEqual(json.dumps(left_payload, sort_keys=True), json.dumps(right_payload, sort_keys=True))
        self.assertEqual(set(ATOMS), {name for name, _ in trace.states[0].predicates})


if __name__ == "__main__":
    unittest.main()
