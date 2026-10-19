from decimal import Decimal
import unittest

from dumbocracy.campaign import ActionContext, CampaignAction, ResourceLedger, commit_action, validate_action
from dumbocracy.config import ActionConfig
from dumbocracy.errors import ErrorCode


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ActionContext(("v0", "v1"), ResourceLedger.create(Decimal("2.00")))
        self.config = ActionConfig(.2, .1, Decimal("1.00"), Decimal("5.00"))

    def test_targeted_commit_is_exact_and_immutable(self) -> None:
        action = CampaignAction("targeted_contact", ("v1",))
        validation = validate_action(self.context, action, self.config)
        self.assertTrue(validation.accepted)
        self.assertEqual(self.context.ledger.remaining_budget, Decimal("2.00"))
        commit = commit_action(self.context, validation)
        self.assertEqual(commit.context.ledger.remaining_budget, Decimal("1.00"))
        self.assertEqual(commit.context.ledger.spent, Decimal("1.00"))
        self.assertEqual(commit.contributions, (0.0, .2))

    def test_rejections_and_noop_do_not_mutate(self) -> None:
        before = self.context
        duplicate = validate_action(before, CampaignAction("targeted_contact", ("v0", "v0")), self.config)
        poor = validate_action(before, CampaignAction("broadcast"), self.config)
        self.assertEqual(duplicate.error_code, ErrorCode.INVALID_TARGET)
        self.assertEqual(poor.error_code, ErrorCode.INSUFFICIENT_BUDGET)
        self.assertEqual(before, self.context)
        no_op = validate_action(before, CampaignAction("no_op"), self.config)
        self.assertTrue(no_op.accepted)
        self.assertEqual(commit_action(before, no_op).context.ledger, before.ledger)

    def test_broadcast_guard_and_unknown_action_are_atomic(self) -> None:
        rich = ActionContext(("v0", "v1"), ResourceLedger.create(Decimal("10.00")))
        accepted = commit_action(rich, validate_action(rich, CampaignAction("broadcast"), self.config))
        repeated = validate_action(accepted.context, CampaignAction("broadcast"), self.config)
        unknown = validate_action(accepted.context, CampaignAction("not_an_action"), self.config)
        self.assertEqual(repeated.error_code, ErrorCode.INVALID_ACTION)
        self.assertEqual(unknown.error_code, ErrorCode.INVALID_ACTION)
        self.assertEqual(accepted.context.ledger.remaining_budget, Decimal("5.00"))
