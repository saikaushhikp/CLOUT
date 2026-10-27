import unittest
from dumbocracy.graph_experiments import ring_lattice_graph


class RingLatticeTests(unittest.TestCase):
    def test_regular_sparse_repeatable(self):
        a = ring_lattice_graph(1000, 8, 42)
        b = ring_lattice_graph(1000, 8, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a.edges), 4000)
        self.assertEqual(a.summary().min_degree, 8)
        self.assertEqual(a.summary().max_degree, 8)
        self.assertEqual(a.summary().component_count, 1)

    def test_small_wraparound(self):
        graph = ring_lattice_graph(6, 2)
        self.assertEqual(len(graph.edges), 6)

    def test_rejects_odd_or_impossible_degree(self):
        for n, k in ((3, 1), (4, 4), (8, 3)):
            with self.assertRaises(ValueError):
                ring_lattice_graph(n, k)


if __name__ == "__main__":
    unittest.main()
