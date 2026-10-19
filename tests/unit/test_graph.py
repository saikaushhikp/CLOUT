from pathlib import Path
import unittest

from dumbocracy.errors import DumbocracyError
from dumbocracy.graph import ElectorateGraph, erdos_renyi_graph, load_edge_list_fixture


ROOT = Path(__file__).resolve().parents[2]


class GraphTests(unittest.TestCase):
    def test_fixture_mapping_and_summary(self) -> None:
        graph = load_edge_list_fixture(ROOT / "tests/fixtures/tiny_path_graph.json")
        self.assertEqual(graph.node_index, {"v0": 0, "v1": 1, "v2": 2})
        self.assertEqual(graph.edges, ((0, 1), (1, 2)))
        self.assertEqual(graph.summary().component_count, 1)
        self.assertEqual(graph.summary().max_degree, 2)

    def test_invalid_edges_and_boundaries(self) -> None:
        with self.assertRaises(DumbocracyError):
            ElectorateGraph.from_edge_ids(("a",), (("a", "a"),))
        with self.assertRaises(DumbocracyError):
            ElectorateGraph.from_edge_ids(("a", "b"), (("a", "b"), ("b", "a")))
        with self.assertRaises(DumbocracyError):
            ElectorateGraph.from_edge_ids(("a",), (("a", "missing"),))
        self.assertEqual(erdos_renyi_graph(4, 0, 3).edges, ())
        self.assertEqual(len(erdos_renyi_graph(4, 1, 3).edges), 6)
        self.assertEqual(erdos_renyi_graph(20, 0.1, 9), erdos_renyi_graph(20, 0.1, 9))
        self.assertEqual(ElectorateGraph(("only",), ()).summary().component_count, 1)

    def test_moderate_sparse_graph_remains_edge_list_backed(self) -> None:
        graph = erdos_renyi_graph(1000, .002, 44)
        self.assertLess(len(graph.edges), 10_000)
        self.assertEqual(graph.summary().node_count, 1000)
