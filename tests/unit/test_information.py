"""Recursive privacy and detachment tests for the public observation boundary."""
import json
import unittest
from decimal import Decimal

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine
from dumbocracy.information import public_json
from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState


PRIVATE_KEYS = {"voters", "support", "susceptibility", "seed", "seeds", "root_seed", "graph_seed", "initial_state_seed", "transition_seed", "strategy_seed", "gamestate", "diagnostics", "contributions", "effects", "support_values"}


def keys_recursively(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key.lower()
            yield from keys_recursively(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from keys_recursively(nested)


class InformationTests(unittest.TestCase):
    def setUp(self):
        self.graph = ElectorateGraph.from_edge_ids(("a", "b"), (("a", "b"),))
        self.config = GameConfig(SeedConfig(4), GraphSpec("edge_list", 2), 2, Decimal("10.00"), DynamicsConfig(.5), ActionConfig(.1, .1))
        self.engine = GameEngine()

    def test_recursive_serialization_has_no_hidden_fields(self):
        state = self.engine.initialize(self.config, self.graph, (VoterState("a", .2, .7), VoterState("b", .8, .3)))
        payload = self.engine.observe(state).to_public_dict()
        keys = set(keys_recursively(payload))
        self.assertFalse(keys & PRIVATE_KEYS)
        self.assertIn("aggregate_support", keys)
        json.loads(public_json(self.engine.observe(state)))
        observation = self.engine.observe(state)
        self.assertIsNot(observation, state)
        self.assertIsNot(observation.public_events, state.internal_events)
        self.assertIsNot(getattr(observation, "voters", None), state.voters)

    def test_observation_nested_mutation_is_detached(self):
        state = self.engine.initialize(self.config, self.graph, (VoterState("a", .2, .7), VoterState("b", .8, .3)))
        baseline = self.engine.observe(state).to_public_dict()
        exposed = self.engine.observe(state).to_public_dict()
        exposed["legal_actions"][0]["parameters"]["delta_target"] = 99
        exposed["graph_summary"]["node_count"] = 999
        self.assertEqual(self.engine.observe(state).to_public_dict(), baseline)

    def test_private_states_with_same_public_aggregate_match_payload(self):
        left = self.engine.initialize(self.config, self.graph, (VoterState("a", .2, .1), VoterState("b", .8, .9)))
        right = self.engine.initialize(self.config, self.graph, (VoterState("a", .4, .9), VoterState("b", .6, .1)))
        self.assertEqual(self.engine.observe(left).to_public_dict(), self.engine.observe(right).to_public_dict())

    def test_event_payload_never_contains_target_ids_or_private_state(self):
        state = self.engine.initialize(self.config, self.graph, (VoterState("a", .2, .1), VoterState("b", .8, .9)))
        result = self.engine.step(state, CampaignAction("targeted_contact", ("a",)))
        payload = result.observation.to_public_dict()
        self.assertIn("a", payload["legal_actions"][0]["eligible_target_ids"])
        self.assertNotIn("target_ids", set(keys_recursively(payload)))
        self.assertNotIn("susceptibility", set(keys_recursively(payload)))
        for event in payload["public_events"]:
            self.assertNotIn("targets", event)
            self.assertNotIn("attributes", event)
