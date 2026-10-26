import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tdindp_capacities import build_capacity_frontier
from tdindp_capacities.generation import _capacity_model, _set_fairness_objective
from tdindp_capacities.networks import Networks


def small_network():
    arcs = [("a", "b"), ("b", "a"), ("a", "b")]
    keys = [(i, "test", "test", 0) for i in ("a", "b")]
    return Networks(
        K=["test"], T=[0], N_k={"test": ["a", "b"]},
        A_k={"test": arcs}, L_k={"test": ["test"]},
        b=dict(zip(keys, (10.0, -10.0))),
        M_plus=dict.fromkeys(keys, 1000.0), M_minus=dict.fromkeys(keys, 1000.0),
        c={(i, j, "test", "test", 0): 1.0 for i, j in arcs},
        u_nominal={}, f={}, q={},
    )


def fake_model(values, status=2, objective=0):
    model = MagicMock()
    model.__enter__.return_value = model
    model.Status = status
    model.ObjVal = objective
    model.getVars.return_value = [
        SimpleNamespace(VarName="u_('a', 'b', 'test', 0)", X=value)
        for value in values
    ]
    return model


class GenerationTests(unittest.TestCase):
    def test_reader_does_not_import_gurobi(self):
        subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.modules['gurobipy'] = None; "
             "from tdindp_capacities import load_frontier; "
             "assert load_frontier().n_points == 19"],
            check=True,
        )

    def test_existing_frontier_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "existing.csv")
            with open(path, "w") as stream:
                stream.write("keep this")
            with self.assertRaises(FileExistsError):
                build_capacity_frontier(small_network(), directory, "existing.csv")
            with open(path) as stream:
                self.assertEqual(stream.read(), "keep this")

    def test_invalid_schedule_or_unsupported_horizon_fails_before_solving(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._capacity_model") as model:
                for arguments in (
                    {"epsilon": 0}, {"epsilon": float("nan")},
                    {"epsilon_decay": 1}, {"epsilon_floor": 1},
                ):
                    with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                        build_capacity_frontier(small_network(), directory, **arguments)
                network = small_network()
                network.T = [0, 1]
                with self.assertRaisesRegex(ValueError, "T ="):
                    build_capacity_frontier(network, directory)
                model.assert_not_called()
            self.assertEqual(os.listdir(directory), [])

    def test_source_schedule_and_objective_bookkeeping_are_retained(self):
        class Budget:
            def __le__(self, value):
                return value

        solver = SimpleNamespace(
            GRB=SimpleNamespace(OPTIMAL=2, INFEASIBLE=3, INF_OR_UNBD=4),
            quicksum=lambda terms: Budget(),
        )
        models = [
            fake_model([2, 10], objective=20),
            fake_model([3, 100]),
            fake_model([4, 70]),
            fake_model([], status=3),
            fake_model([2, 60]),
        ]
        network = small_network()
        network.A_k["test"] = [("a", "b"), ("a", "b")]
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model", side_effect=models) as build, \
                    patch("tdindp_capacities.generation._set_fairness_objective") as fairness, \
                    self.assertLogs("tdindp_capacities.generation", level="WARNING"):
                frontier = build_capacity_frontier(
                    network, directory, epsilon_floor=0.98
                )
            self.assertEqual(build.call_count, 5)
            for call in fairness.call_args_list:
                self.assertEqual(call.args[2], {"test": 10})
            for model, budget in zip(models[2:], (100, 74 * 0.99, 74 * 0.99**2)):
                self.assertAlmostEqual(model.addConstr.call_args.args[0], budget)
            self.assertEqual(frontier.n_points, 3)
            self.assertEqual(frontier.capacities(0)[("a", "b", "test", 0)], 10)
            self.assertEqual(frontier.objectives["sum_u"].iloc[0], 20)
            self.assertAlmostEqual(frontier.objectives["sum_log_u"].iloc[0], math.log(10))
            self.assertEqual(frontier.objectives["sum_u"].iloc[1], 74)
            self.assertAlmostEqual(
                frontier.objectives["sum_log_u"].iloc[1],
                math.log(4 + 1e-6) + math.log(70 + 1e-6),
            )

    def test_failed_anchor_does_not_create_output(self):
        solver = SimpleNamespace(GRB=SimpleNamespace(OPTIMAL=2))
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model",
                          return_value=fake_model([], status=3)):
                with self.assertRaisesRegex(RuntimeError, "anchor failed"):
                    build_capacity_frontier(small_network(), directory)
            self.assertEqual(os.listdir(directory), [])


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is optional")
class FormulationTests(unittest.TestCase):
    def test_source_variables_constraints_and_defaults_are_retained(self):
        import gurobipy as gp

        network = small_network()
        with _capacity_model(network) as model:
            variables = model.getVars()
            self.assertEqual(model.NumBinVars, 5)
            self.assertEqual(model.NumGenConstrs, 3)
            self.assertEqual(model.NumQConstrs, 9)
            duplicate_name = "u_('a', 'b', 'test', 0)"
            duplicate_vars = [v for v in variables if v.VarName == duplicate_name]
            self.assertEqual(len(duplicate_vars), 2)
            self.assertEqual([v.Obj for v in duplicate_vars], [0, 2])
            self.assertTrue(all(v.LB == 0 for v in duplicate_vars))
            self.assertTrue(all(v.UB >= gp.GRB.INFINITY for v in duplicate_vars))
            self.assertEqual(model.Params.FuncNonlinear, model.getParamInfo("FuncNonlinear")[-1])
            self.assertEqual(model.Params.FeasibilityTol, model.getParamInfo("FeasibilityTol")[-1])
            self.assertIsNotNone(model.getConstrByName("meet_demand_('b', 'test', 'test', 0)"))
            excess = model.getVarByName("delta_plus_('a', 'test', 'test', 0)")
            self.assertGreaterEqual(excess.UB, gp.GRB.INFINITY)
            self.assertEqual(excess.Obj, 0)

            _set_fairness_objective(model, network, {"test": 10})
            log_variable = model.getVarByName("log_u_('a', 'b', 'test', 0)")
            self.assertEqual(log_variable.Obj, 2)
            self.assertGreaterEqual(log_variable.UB, gp.GRB.INFINITY)
            bounds = [c for c in model.getConstrs() if c.ConstrName.startswith("max_cap_")]
            self.assertEqual(len(bounds), 3)
            self.assertTrue(all(abs(c.RHS - math.log(10)) < 1e-12 for c in bounds))


if __name__ == "__main__":
    unittest.main()
