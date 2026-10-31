import hashlib
import importlib.util
import math
import os
import unittest

import pandas as pd

from tdindp_capacities import (
    BENCHMARK_FILENAME, PRE_FIX_DATA_DIR, Gamma, load_benchmark, load_capacities,
    load_frontier, load_networks,
)
from tdindp_capacities.generation import _capacity_model


REPEATED_ARC = ("Power_node_5", "Power_node_64")


class GammaTests(unittest.TestCase):
    def setUp(self):
        self.gamma = load_networks().gamma
        self.key = next(iter(self.gamma))

    def test_malformed_lookups_raise(self):
        for key in (self.key[:4], self.key + (0,), "Gas_node_0", [*self.key]):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "gamma keys"):
                    key in self.gamma
                with self.assertRaisesRegex(ValueError, "gamma keys"):
                    self.gamma[key]
                with self.assertRaisesRegex(ValueError, "gamma keys"):
                    self.gamma.get(key)

    def test_well_formed_lookups_behave_like_a_dict(self):
        absent = ("x", "y", "Gas", "Power", 0)
        self.assertIn(self.key, self.gamma)
        self.assertEqual(self.gamma[self.key], 1)
        self.assertEqual(self.gamma.get(self.key), 1)
        self.assertNotIn(absent, self.gamma)
        self.assertEqual(self.gamma.get(absent, "default"), "default")
        with self.assertRaises(KeyError):
            self.gamma[absent]

    def test_loader_keeps_every_dependency(self):
        self.assertIsInstance(self.gamma, Gamma)
        self.assertEqual(len(self.gamma), 27)
        self.assertEqual(len(list(self.gamma.items())), 27)


class RepeatedArcTests(unittest.TestCase):
    def setUp(self):
        self.networks = load_networks()

    def test_every_layer_lists_each_arc_once(self):
        for k in self.networks.K:
            self.assertEqual(len(self.networks.A_k[k]), len(set(self.networks.A_k[k])))
        self.assertEqual(self.networks.A_k["Power"].count(REPEATED_ARC), 1)
        self.assertEqual(self.networks.summary()["arcs"].tolist(), [34, 184, 142])

    def test_arcs_match_the_frontier_columns(self):
        self.assertEqual(len(self.networks.arc_keys()), 360)
        self.assertEqual(set(self.networks.arc_keys()), set(load_frontier().arc_keys))
        self.assertEqual(set(self.networks.arc_keys()), set(load_benchmark().arc_keys))

    def test_repeated_row_keeps_its_later_values_and_its_reverse(self):
        i, j = REPEATED_ARC
        self.assertIn((j, i), self.networks.A_k["Power"])
        for key in ((i, j, "Power", 0), (j, i, "Power", 0)):
            self.assertEqual(self.networks.f[key], 14274.0)
        self.assertEqual(self.networks.c[(i, j, "Power", "Power", 0)], 14274.0)


class CorrectedFrontierTests(unittest.TestCase):
    def setUp(self):
        self.frontier = load_frontier()

    def assert_matches_capacities(self, row, sum_u, sum_log_u):
        values = list(self.frontier.capacities(row).values())
        self.assertEqual(sum_u, sum(values))
        self.assertEqual(sum_log_u, sum(math.log(v) for v in values))

    def test_default_reader_uses_the_corrected_frontier(self):
        self.assertEqual(
            os.path.basename(self.frontier.path), "pareto_regenerated_2026-10-31.csv"
        )
        self.assertEqual((self.frontier.n_points, self.frontier.n_arcs), (19, 360))
        self.assertEqual(load_capacities(), self.frontier.capacities(-1))
        self.assertAlmostEqual(self.frontier.point_summary(0)["sum_u"], 10048.787462, places=5)
        self.assertAlmostEqual(self.frontier.point_summary(18)["sum_u"], 10818.395348, places=5)
        self.assertAlmostEqual(self.frontier.point_summary(18)["sum_log_u"], 1056.963372, places=5)

    def test_summaries_are_recomputed_for_every_row(self):
        for row in range(self.frontier.n_points):
            summary = self.frontier.point_summary(row)
            self.assert_matches_capacities(row, summary["sum_u"], summary["sum_log_u"])
        for _, line in self.frontier.summary_table().iterrows():
            self.assert_matches_capacities(int(line["row"]), line["sum_u"], line["sum_log_u"])

    def test_stored_objectives_reconcile(self):
        check = self.frontier.objective_consistency()
        self.assertLess(check["sum_u_error"].abs().max(), 1e-6)
        self.assertLess(check["sum_log_u_error"].abs().max(), 1e-6)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = load_benchmark()

    def test_benchmark_is_the_archived_file(self):
        self.assertEqual(os.path.dirname(self.benchmark.path), PRE_FIX_DATA_DIR)
        self.assertEqual(os.path.basename(self.benchmark.path), BENCHMARK_FILENAME)
        self.assertEqual((self.benchmark.n_points, self.benchmark.n_arcs), (19, 360))

    def test_published_values_are_unchanged(self):
        self.assertAlmostEqual(self.benchmark.point_summary(0)["sum_u"], 10048.78746, places=4)
        self.assertAlmostEqual(self.benchmark.point_summary(18)["sum_u"], 11440.774734, places=5)
        self.assertAlmostEqual(self.benchmark.point_summary(18)["sum_log_u"], 1114.22537, places=4)

    def test_snapshots_hold_the_same_published_file(self):
        def digest(path):
            with open(path, "rb") as stream:
                return hashlib.sha256(stream.read()).hexdigest()

        root = os.path.dirname(PRE_FIX_DATA_DIR)
        for snapshot in ("code_published_frontier", "code_dependency_fix"):
            self.assertEqual(
                digest(os.path.join(root, snapshot, "data", "pareto_of_capacities.csv")),
                digest(self.benchmark.path),
            )

    def test_stored_objectives_still_show_the_repeated_arc(self):
        check = self.benchmark.objective_consistency()
        self.assertAlmostEqual(check["sum_u_error"].iloc[0], -11.9711, places=3)
        pd.testing.assert_series_equal(
            check["sum_log_u_error"].iloc[1:],
            pd.Series([-math.log(1e-6)] * 18, index=range(1, 19), name="sum_log_u_error"),
            atol=1e-4,
        )


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is optional")
class LoadedModelTests(unittest.TestCase):
    def test_loaded_instance_has_one_capacity_variable_per_arc(self):
        with _capacity_model(load_networks()) as model:
            names = [v.VarName for v in model.getVars() if v.VarName[0] == "u"]
            self.assertEqual(len(names), 360)
            self.assertEqual(len(set(names)), 360)
            variable = model.getVarByName("u_" + str(REPEATED_ARC + ("Power", 0)))
            self.assertEqual(variable.Obj, 1)


if __name__ == "__main__":
    unittest.main()
