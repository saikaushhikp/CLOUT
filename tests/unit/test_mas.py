import unittest

from dumbocracy.graph import ElectorateGraph
from dumbocracy.mas import VoterState, advance_support, player_vote_share
from dumbocracy.errors import DumbocracyError, ErrorCode


class MasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = ElectorateGraph.from_edge_ids(("v0", "v1", "v2"), (("v0", "v1"), ("v1", "v2")))
        self.voters = (VoterState("v0", .2, .5), VoterState("v1", .5, .5), VoterState("v2", .8, .5))

    def test_hand_computed_synchronous_path(self) -> None:
        # Old means: [.5, .5, .5], so an in-place implementation would differ at v1/v2.
        result = advance_support(self.graph, self.voters, .5, (.2, 0, 0))
        self.assertEqual([round(v.support, 6) for v in result], [.375, .5, .725])
        self.assertEqual([v.support for v in self.voters], [.2, .5, .8])

    def test_isolate_endpoints_clipping_and_threshold(self) -> None:
        isolate = ElectorateGraph(("solo",), ())
        voter = (VoterState("solo", .2, 1),)
        self.assertEqual(advance_support(isolate, voter, 1, (0,))[0].support, 0)
        self.assertEqual(advance_support(isolate, voter, 0, (2,))[0].support, 1)
        self.assertEqual(advance_support(isolate, voter, 0, (-2,))[0].support, 0)
        self.assertEqual(player_vote_share((VoterState("a", .5, 0), VoterState("b", 1, 0))), .5)

    def test_nonfinite_contribution_fails_without_changing_input(self) -> None:
        with self.assertRaises(DumbocracyError) as caught:
            advance_support(self.graph, self.voters, .5, (float("inf"), 0, 0))
        self.assertEqual(caught.exception.code, ErrorCode.NUMERICAL_INSTABILITY)
        self.assertEqual([v.support for v in self.voters], [.2, .5, .8])
