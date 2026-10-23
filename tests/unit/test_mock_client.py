"""The mock client consumes only serialized public observation data."""
import unittest
import json
from pathlib import Path
import subprocess
import sys

from examples.week3_mock_client import choose_action


class MockClientTests(unittest.TestCase):
    def test_safe_public_observation_fixture_and_action_shape(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "public_observation_v1.json"
        observation = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(observation["schema_version"], "1.0")
        self.assertTrue({"round_index", "remaining_budget", "aggregate_support", "graph_summary", "legal_actions", "terminal_status", "public_events"} <= observation.keys())
        payload = choose_action(observation)
        self.assertEqual(set(payload), {"action_id", "target_ids"})
        self.assertEqual(payload["action_id"], "targeted_contact")
        self.assertIn(payload["target_ids"][0], observation["legal_actions"][0]["eligible_target_ids"])

    def test_fixture_json_round_trip_and_additive_field_compatibility(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "public_observation_v1.json"
        observation = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(observation)), observation)

        # Schema-compatible additive public data must not change the action
        # contract; the client continues to use only supported fields.
        extended = {**observation, "optional_public_extension": {"label": "v1-compatible"}}
        self.assertEqual(
            choose_action(extended),
            {"action_id": "targeted_contact", "target_ids": ["v0"]},
        )

    def test_cli_reads_public_fixture_and_emits_json_action(self):
        root = Path(__file__).resolve().parents[2]
        fixture = root / "tests" / "fixtures" / "public_observation_v1.json"
        client = root / "examples" / "week3_mock_client.py"
        result = subprocess.run(
            [sys.executable, str(client)],
            input=fixture.read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"action_id": "targeted_contact", "target_ids": ["v0"]},
        )

    def test_selects_first_public_eligible_target(self):
        observation = {"legal_actions": [
            {"action_id": "targeted_contact", "eligible_target_ids": ["v0", "v1"]},
            {"action_id": "no_op", "eligible_target_ids": []},
        ]}
        self.assertEqual(choose_action(observation), {"action_id": "targeted_contact", "target_ids": ["v0"]})

    def test_uses_no_op_when_targeted_action_is_unavailable(self):
        observation = {"legal_actions": [{"action_id": "no_op", "eligible_target_ids": []}]}
        self.assertEqual(choose_action(observation), {"action_id": "no_op", "target_ids": []})

    def test_malformed_payload_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            choose_action({"legal_actions": [None]})
        with self.assertRaises(ValueError):
            choose_action({"legal_actions": [{"action_id": "targeted_contact", "eligible_target_ids": "v0"}]})

    def test_client_has_no_engine_import(self):
        source = Path(__file__).resolve().parents[2] / "examples" / "week3_mock_client.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("from dumbocracy.engine", text)
        self.assertNotIn("import GameState", text)
