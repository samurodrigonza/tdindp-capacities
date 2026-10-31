import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from test_generation import fake_model, small_network
from tdindp_capacities.moving_ceiling import build_moving_ceiling_frontier


class MovingCeilingTests(unittest.TestCase):
    def test_ceilings_follow_the_last_feasible_solve(self):
        class Budget:
            def __le__(self, value):
                return value

        solver = SimpleNamespace(
            GRB=SimpleNamespace(OPTIMAL=2, INFEASIBLE=3, INF_OR_UNBD=4),
            quicksum=lambda terms: Budget(),
            gurobi=SimpleNamespace(version=lambda: (11, 0, 3)),
        )
        models = [
            fake_model([10], objective=10),
            fake_model([100]),
            fake_model([70]),
            fake_model([], status=3),
            fake_model([60]),
            fake_model([50]),
        ]
        network = small_network()
        network.A_k["test"] = [("a", "b")]
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.moving_ceiling._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.moving_ceiling._capacity_model", side_effect=models), \
                    patch("tdindp_capacities.moving_ceiling._set_fairness_objective") as fairness:
                frontier = build_moving_ceiling_frontier(
                    network, directory, "out.csv", epsilon_floor=0.965
                )
            # Fairness anchor and first sweep share the minimum solve's ceiling.
            self.assertEqual(
                [call.args[2]["test"] for call in fairness.call_args_list],
                [10, 10, 70, 70, 60],
            )
            for model, budget in zip(models[2:], (100, 70 * 0.99, 70 * 0.99**2, 60 * 0.99**3)):
                self.assertAlmostEqual(model.addConstr.call_args.args[0], budget)
            self.assertEqual(frontier.n_points, 4)
            trace = pd.read_csv(Path(directory) / "out_trace.csv")
            self.assertEqual(
                trace["outcome"].tolist(), ["accepted", "infeasible", "accepted", "accepted"]
            )
            self.assertEqual(trace["ceiling_test"].tolist(), [10, 70, 70, 60])
            self.assertEqual(frontier.objectives["sum_u"].tolist(), [10, 70, 60, 50])

    def test_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("out.csv", "out_trace.csv"):
                path = os.path.join(directory, name)
                with open(path, "w") as stream:
                    stream.write("keep this")
                with self.assertRaises(FileExistsError):
                    build_moving_ceiling_frontier(small_network(), directory, "out.csv")
                with open(path) as stream:
                    self.assertEqual(stream.read(), "keep this")
                os.remove(path)

    def test_invalid_inputs_fail_before_solving(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.moving_ceiling._capacity_model") as model:
                for arguments in (
                    {"epsilon": 0}, {"epsilon_decay": 1}, {"epsilon_floor": 1},
                    {"func_nonlinear": 2},
                ):
                    with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                        build_moving_ceiling_frontier(small_network(), directory, **arguments)
                repeated = small_network()
                repeated.A_k["test"].append(("a", "b"))
                with self.assertRaisesRegex(ValueError, "unique arcs"):
                    build_moving_ceiling_frontier(repeated, directory)
                model.assert_not_called()
            self.assertEqual(os.listdir(directory), [])


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is optional")
class MovingCeilingSolverTests(unittest.TestCase):
    def test_real_sweep_never_exceeds_its_falling_ceilings(self):
        network = small_network()
        network.N_k["test"].append("c")
        network.b[("c", "test", "test", 0)] = 0
        network.A_k["test"] = [
            ("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("c", "b"), ("b", "c"),
        ]
        network.c = {(i, j, "test", "test", 0): 1.0 for i, j in network.A_k["test"]}
        with tempfile.TemporaryDirectory() as directory:
            frontier = build_moving_ceiling_frontier(
                network, directory, "out.csv", epsilon_floor=0.9, func_nonlinear=1
            )
            trace = pd.read_csv(Path(directory) / "out_trace.csv")
        self.assertGreater(frontier.n_points, 2)
        ceilings = trace["ceiling_test"]
        self.assertTrue(((ceilings.diff().dropna()) <= 1e-6).all())
        accepted = trace[trace["outcome"] == "accepted"]
        self.assertEqual(len(accepted), frontier.n_points - 1)
        for row, ceiling in enumerate(accepted["ceiling_test"], start=1):
            self.assertLessEqual(max(frontier.capacities(row).values()), ceiling + 1e-4)


if __name__ == "__main__":
    unittest.main()
