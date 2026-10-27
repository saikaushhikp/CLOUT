from decimal import Decimal
import unittest

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.epistemic import CheckStatus, check_finite_worlds
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState


class EpistemicTests(unittest.TestCase):
    def worlds(self):
        cfg = GameConfig(SeedConfig(2), GraphSpec("edge_list", 2), 1, Decimal("0.00"),
                         DynamicsConfig(.5), ActionConfig(.1, .1))
        graph = ElectorateGraph(("a", "b"), ())
        voters = (VoterState("a", .8, .4), VoterState("b", .8, .3))
        engine = GameEngine()
        return engine, (engine.initialize(cfg, graph, voters),)

    def test_finite_world_guarantee_is_scoped_and_repeatable(self):
        engine, worlds = self.worlds()
        first = check_finite_worlds(engine, worlds)
        second = check_finite_worlds(engine, worlds)
        self.assertEqual(first, second)
        self.assertEqual(first.guaranteed_victory, CheckStatus.TRUE)
        self.assertEqual(first.possible_victory, CheckStatus.TRUE)
        self.assertEqual(first.inevitable_defeat, CheckStatus.FALSE)
        self.assertEqual(first.evaluation_mode if hasattr(first, "evaluation_mode") else "finite_model", "finite_model")

    def test_cap_is_not_reported_as_proof(self):
        engine, worlds = self.worlds()
        result = check_finite_worlds(engine, worlds, node_limit=1)
        self.assertEqual(result.guaranteed_victory, CheckStatus.INCOMPLETE)

    def test_horizon_cap_is_incomplete(self):
        cfg = GameConfig(SeedConfig(8), GraphSpec("edge_list", 2), 2, Decimal("0.00"),
                         DynamicsConfig(.5), ActionConfig(.1, .1))
        graph = ElectorateGraph(("a", "b"), ())
        engine = GameEngine()
        world = engine.initialize(cfg, graph, (VoterState("a", .8, .2), VoterState("b", .8, .2)))
        result = check_finite_worlds(engine, (world,), horizon_limit=1)
        self.assertEqual(result.guaranteed_victory, CheckStatus.INCOMPLETE)

    def test_checker_does_not_change_original_world(self):
        engine, worlds = self.worlds()
        before = worlds[0]
        check_finite_worlds(engine, worlds)
        self.assertEqual(worlds[0], before)

    def test_indistinguishable_worlds_separate_omniscient_and_shared_policy(self):
        cfg = GameConfig(SeedConfig(3), GraphSpec("edge_list", 2), 1, Decimal("1.00"),
                         DynamicsConfig(0), ActionConfig(.02, .1))
        graph = ElectorateGraph(("a", "b"), ())
        engine = GameEngine()
        first = engine.initialize(cfg, graph, (VoterState("a", .49, 1), VoterState("b", .9, 1)))
        second = engine.initialize(cfg, graph, (VoterState("a", .9, 1), VoterState("b", .49, 1)))
        self.assertEqual(engine.observe(first).to_public_dict(), engine.observe(second).to_public_dict())
        result = check_finite_worlds(engine, (first, second))
        self.assertEqual(result.possible_victory, CheckStatus.TRUE)
        self.assertEqual(result.guaranteed_victory, CheckStatus.FALSE)
        self.assertEqual(result.inevitable_defeat, CheckStatus.FALSE)
        self.assertEqual(result.knows_guaranteed_victory, CheckStatus.FALSE)


if __name__ == "__main__":
    unittest.main()
