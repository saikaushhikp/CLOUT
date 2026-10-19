"""End-to-end path episode tests for later user/reviewer execution."""
from decimal import Decimal
import unittest

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import load_edge_list_fixture
from dumbocracy.mas import VoterState
from pathlib import Path


class EpisodeTests(unittest.TestCase):
    def test_fixture_episode_actions_replay_and_result(self):
        root = Path(__file__).resolve().parents[2]
        graph = load_edge_list_fixture(root / "tests/fixtures/tiny_path_graph.json")
        voters = (VoterState("v0", .2, .5), VoterState("v1", .5, .5), VoterState("v2", .8, .5))
        config = GameConfig(SeedConfig(20261019), GraphSpec("edge_list", 3), 2, Decimal("10.00"), DynamicsConfig(.5), ActionConfig(.2, .1))
        engine = GameEngine(revision="test")
        state = engine.initialize(config, graph, voters)
        initial_observation = engine.observe(state)
        self.assertEqual(initial_observation.aggregate_support, .5)
        first = engine.step(state, CampaignAction("targeted_contact", ("v0",)))
        self.assertEqual([round(v.support, 6) for v in first.state.voters], [.375, .5, .725])
        second = engine.step(first.state, CampaignAction("no_op"))
        final = second.state
        self.assertEqual(final.round_index, 2)
        self.assertTrue(final.terminal)
        self.assertEqual(final.result.player_votes, 2)
        self.assertEqual(final.result.vote_share, 2 / 3)
        self.assertEqual(final.ledger.remaining_budget, Decimal("9.00"))
