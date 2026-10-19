"""The mock client consumes only serialized public observation data."""
import unittest

from examples.week3_mock_client import choose_action


class MockClientTests(unittest.TestCase):
    def test_selects_first_public_eligible_target(self):
        observation = {"legal_actions": [
            {"action_id": "targeted_contact", "eligible_target_ids": ["v0", "v1"]},
            {"action_id": "no_op", "eligible_target_ids": []},
        ]}
        self.assertEqual(choose_action(observation), {"action_id": "targeted_contact", "target_ids": ["v0"]})

    def test_uses_no_op_when_targeted_action_is_unavailable(self):
        observation = {"legal_actions": [{"action_id": "no_op", "eligible_target_ids": []}]}
        self.assertEqual(choose_action(observation), {"action_id": "no_op", "target_ids": []})
