import unittest
from decimal import Decimal
from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.epistemic import CheckStatus
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState
from dumbocracy.verification import verify_episode


class VerificationTests(unittest.TestCase):
    def completed(self):
        cfg = GameConfig(SeedConfig(5), GraphSpec("edge_list", 2), 1,
                         Decimal("0.00"), DynamicsConfig(0), ActionConfig(.1, .1))
        graph = ElectorateGraph(("a", "b"), ())
        engine = GameEngine()
        state = engine.initialize(cfg, graph, (VoterState("a", .8, .5), VoterState("b", .8, .5)))
        terminal = engine.step(state, CampaignAction("no_op")).state
        return engine, state, terminal

    def test_trace_and_missing_finite_model_are_distinguished(self):
        engine, initial, terminal = self.completed()
        report = verify_episode(terminal, engine)
        self.assertEqual(report["trace"]["evaluation_mode"], "trace_only")
        self.assertEqual(report["finite_model"]["status"], "NOT_SUPPLIED")

    def test_supplied_worlds_add_scoped_finite_results(self):
        engine, initial, terminal = self.completed()
        report = verify_episode(terminal, engine, (initial,))
        self.assertEqual(report["finite_model"]["evaluation_mode"], "finite_model")
        self.assertEqual(report["finite_model"]["guaranteed_victory"], CheckStatus.TRUE.value)


if __name__ == "__main__":
    unittest.main()
