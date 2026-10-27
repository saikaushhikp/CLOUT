import unittest
from decimal import Decimal
from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState
from dumbocracy.verification import verify_episode


class EpistemicEpisodeIntegrationTests(unittest.TestCase):
    def test_actual_engine_episode_and_initial_world_are_separately_scoped(self):
        cfg = GameConfig(SeedConfig(7), GraphSpec("edge_list", 2), 1,
                         Decimal("0.00"), DynamicsConfig(0), ActionConfig(.1, .1))
        graph = ElectorateGraph(("a", "b"), ())
        engine = GameEngine()
        world = engine.initialize(cfg, graph,
                                  (VoterState("a", .8, .2), VoterState("b", .9, .2)))
        completed = engine.step(world, CampaignAction("no_op")).state
        report = verify_episode(completed, engine, (world,))
        self.assertEqual(report["trace"]["evaluation_mode"], "trace_only")
        self.assertEqual(report["finite_model"]["evaluation_mode"], "finite_model")
        self.assertEqual(report["finite_model"]["world_count"], 1)


if __name__ == "__main__":
    unittest.main()
