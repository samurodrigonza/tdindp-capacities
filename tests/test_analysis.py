import importlib.util
import unittest

from test_generation import small_network
from tdindp_capacities import load_networks
from tdindp_capacities.analysis import (
    layer_bottlenecks, layer_throughput_bounds, net_flow_ranges, recheck_capacities,
    solve_under_ceilings, undirected_connections,
)


def triangle_network():
    """Node a supplies 10 to b, directly or through c, with unique arcs."""
    network = small_network()
    network.N_k["test"].append("c")
    network.b[("c", "test", "test", 0)] = 0
    network.A_k["test"] = [
        ("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("c", "b"), ("b", "c"),
    ]
    network.c = {(i, j, "test", "test", 0): 1.0 for i, j in network.A_k["test"]}
    return network


class ConnectionAndBoundTests(unittest.TestCase):
    def test_reverse_pairs_count_once(self):
        self.assertEqual(
            undirected_connections(small_network()), {"test": [("a", "b")]}
        )
        counts = {k: len(v) for k, v in undirected_connections(load_networks()).items()}
        self.assertEqual(counts, {"Gas": 17, "Power": 92, "Water": 71})

    def test_throughput_bound_uses_the_tightest_node(self):
        self.assertEqual(
            layer_throughput_bounds(triangle_network())["test"],
            {"bound": 5.0, "node": "a", "availability": 10.0, "degree": 2},
        )

    def test_instance_bounds_explain_the_final_moving_ceilings(self):
        bounds = layer_throughput_bounds(load_networks())
        self.assertAlmostEqual(bounds["Gas"]["bound"], 351.35, places=6)
        self.assertEqual(bounds["Gas"]["node"], "Gas_node_1")
        self.assertAlmostEqual(bounds["Power"]["bound"], 86.0, places=6)
        self.assertEqual(bounds["Power"]["degree"], 3)
        self.assertAlmostEqual(bounds["Water"]["bound"], 55.167, places=6)
        self.assertEqual(bounds["Water"]["node"], "Water_node_16")


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is optional")
class AnalysisSolverTests(unittest.TestCase):
    def test_bottleneck_splits_flow_across_both_routes(self):
        self.assertAlmostEqual(layer_bottlenecks(triangle_network())["test"], 5.0, places=6)

    def test_instance_bottlenecks_match_the_throughput_bounds(self):
        networks = load_networks()
        bounds = layer_throughput_bounds(networks)
        for k, value in layer_bottlenecks(networks).items():
            self.assertAlmostEqual(value, bounds[k]["bound"], places=5)

    def test_ceilings_below_the_bottleneck_are_infeasible(self):
        result = solve_under_ceilings(
            triangle_network(), {"test": 4.0}, minimize_total=True, func_nonlinear=1,
        )
        self.assertIsNone(result["capacities"])

    def test_ceiling_and_budget_solves_respect_their_limits(self):
        network = triangle_network()
        total = solve_under_ceilings(
            network, {"test": 5.0}, minimize_total=True, func_nonlinear=1,
        )
        # Both routes carry 5 at the ceiling: three connections, two directions.
        self.assertAlmostEqual(sum(total["capacities"].values()), 30.0, places=3)
        limited = solve_under_ceilings(network, {"test": 8.0}, budget=30.0, func_nonlinear=1)
        self.assertLessEqual(sum(limited["capacities"].values()), 30.0 + 1e-4)
        self.assertLessEqual(max(limited["capacities"].values()), 8.0 + 1e-4)

    def test_flow_ranges_show_forced_and_free_flow(self):
        network = triangle_network()
        capacities = {(i, j, "test", 0): 5.0 for i, j in network.A_k["test"]}
        forced = net_flow_ranges(network, capacities, "test", [("a", "b")], slack=0)
        self.assertAlmostEqual(forced[("a", "b")][0], 5.0, places=6)
        self.assertAlmostEqual(forced[("a", "b")][1], 5.0, places=6)
        capacities = {(i, j, "test", 0): 10.0 for i, j in network.A_k["test"]}
        free = net_flow_ranges(network, capacities, "test", [("a", "b")], slack=0)
        self.assertAlmostEqual(free[("a", "b")][0], 0.0, places=6)
        self.assertAlmostEqual(free[("a", "b")][1], 10.0, places=6)

    def test_fixing_binaries_at_one_keeps_the_optimum(self):
        network = triangle_network()
        free = solve_under_ceilings(network, {"test": 8.0}, budget=30.0, func_nonlinear=1)
        fixed = solve_under_ceilings(
            network, {"test": 8.0}, budget=30.0, func_nonlinear=1, all_functional=True,
        )
        self.assertAlmostEqual(free["objective"], fixed["objective"], places=3)

    def test_recheck_accepts_a_design_and_rejects_a_mismatch(self):
        network = triangle_network()
        design = solve_under_ceilings(
            network, {"test": 8.0}, budget=30.0, func_nonlinear=1,
        )["capacities"]
        self.assertTrue(recheck_capacities(network, design, func_nonlinear=1))
        broken = dict(design)
        broken[("a", "b", "test", 0)] += 1.0
        self.assertFalse(recheck_capacities(network, broken, func_nonlinear=1))
        self.assertTrue(recheck_capacities(network, broken, allowance=1.0, func_nonlinear=1))


if __name__ == "__main__":
    unittest.main()
