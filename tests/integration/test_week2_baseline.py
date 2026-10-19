from decimal import Decimal
import json
from pathlib import Path
import unittest

from dumbocracy.campaign import ActionContext, CampaignAction, ResourceLedger, commit_action, validate_action
from dumbocracy.config import ActionConfig
from dumbocracy.errors import ErrorCode
from dumbocracy.graph import load_edge_list_fixture
from dumbocracy.mas import VoterState, advance_support


ROOT = Path(__file__).resolve().parents[2]


class Week2BaselineIntegrationTests(unittest.TestCase):
    def test_fixture_action_then_synchronous_update_is_reproducible(self) -> None:
        graph = load_edge_list_fixture(ROOT / "tests/fixtures/tiny_path_graph.json")
        state = json.loads((ROOT / "tests/fixtures/tiny_initial_state.json").read_text())
        voters = tuple(VoterState(**voter) for voter in state["voters"])
        context = ActionContext(graph.node_ids, ResourceLedger.create(Decimal("10.00")))
        config = ActionConfig(.2, .1)
        accepted = validate_action(context, CampaignAction("targeted_contact", ("v0",)), config)
        commit = commit_action(context, accepted)
        after = advance_support(graph, voters, .5, commit.contributions)
        self.assertEqual([round(v.support, 6) for v in after], [.375, .5, .725])
        self.assertEqual(commit.context.ledger.remaining_budget, Decimal("9.00"))
        rejected = validate_action(commit.context, CampaignAction("targeted_contact", ("missing",)), config)
        self.assertEqual(rejected.error_code, ErrorCode.INVALID_TARGET)
        self.assertEqual(commit.context.ledger.remaining_budget, Decimal("9.00"))
