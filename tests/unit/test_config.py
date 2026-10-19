from decimal import Decimal
import math
import unittest

from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.errors import DumbocracyError


class ConfigTests(unittest.TestCase):
    def test_valid_config_and_stable_distinct_subseeds(self) -> None:
        seeds = SeedConfig(7)
        self.assertEqual(seeds.graph, SeedConfig(7).graph)
        self.assertEqual(len({seeds.graph, seeds.initial_state, seeds.transition, seeds.strategy}), 4)
        config = GameConfig(seeds, GraphSpec("erdos_renyi", 3, 0.5), 2, Decimal("10.00"), DynamicsConfig(0.5), ActionConfig(0.2, 0.1))
        self.assertEqual(config.initial_budget, Decimal("10.00"))

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(DumbocracyError): GraphSpec("bad", 2)
        with self.assertRaises(DumbocracyError): GraphSpec("erdos_renyi", 2, math.nan)
        with self.assertRaises(DumbocracyError): DynamicsConfig(1.1)
        with self.assertRaises(DumbocracyError): ActionConfig(math.inf, 0.1)
        with self.assertRaises(DumbocracyError): ActionConfig(0.1, 0.2, Decimal("-1"))
        with self.assertRaises(DumbocracyError): SeedConfig(True)
