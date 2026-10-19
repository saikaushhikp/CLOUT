"""GPU/CPU semantic comparison; CUDA is skipped when unavailable."""
from decimal import Decimal
import unittest

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState


class BackendEquivalenceTests(unittest.TestCase):
    def test_fixed_episode_cpu_and_cuda_match_within_tolerance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA driver/device unavailable")
        graph = ElectorateGraph.from_edge_ids(("a", "b", "c"), (("a", "b"), ("b", "c")))
        config = GameConfig(SeedConfig(123), GraphSpec("edge_list", 3), 2, Decimal("10.00"), DynamicsConfig(.4), ActionConfig(.2, .1))
        voters = (VoterState("a", .1, .4), VoterState("b", .55, .6), VoterState("c", .9, .3))
        actions = (CampaignAction("broadcast"), CampaignAction("no_op"))
        cpu = GameEngine("cpu").replay(config, actions, graph)[-1]
        gpu = GameEngine("cuda:0").replay(config, actions, graph)[-1]
        for left, right in zip(cpu.voters, gpu.voters):
            self.assertAlmostEqual(left.support, right.support, delta=1e-10)
        self.assertEqual(cpu.result.victory, gpu.result.victory)
        self.assertEqual(cpu.ledger, gpu.ledger)
