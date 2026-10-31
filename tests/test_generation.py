import importlib.util
import logging
import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from tdindp_capacities import build_capacity_frontier
from tdindp_capacities.generation import (
    _capacity_model, _optimize_logged, _set_fairness_objective,
)
from tdindp_capacities.networks import Networks


def small_network():
    arcs = [("a", "b"), ("b", "a")]
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
    model.SolCount = int(status == 2)
    model.Params = SimpleNamespace(FuncNonlinear=1, MIPGap=1e-4)
    names = ("u_('a', 'b', 'test', 0)", "u_('b', 'a', 'test', 0)")
    model.getVars.return_value = [
        SimpleNamespace(VarName=name, X=value) for name, value in zip(names, values)
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
                    {"func_nonlinear": 2}, {"func_nonlinear": "nonlinear"},
                ):
                    with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                        build_capacity_frontier(small_network(), directory, **arguments)
                network = small_network()
                network.T = [0, 1]
                with self.assertRaisesRegex(ValueError, "T ="):
                    build_capacity_frontier(network, directory)
                network = small_network()
                network.A_k["test"].append(("a", "b"))
                with self.assertRaisesRegex(ValueError, "unique arcs"):
                    build_capacity_frontier(network, directory)
                model.assert_not_called()
            self.assertEqual(os.listdir(directory), [])

    def test_sweep_schedule_and_objectives(self):
        class Budget:
            def __le__(self, value):
                return value

        solver = SimpleNamespace(
            GRB=SimpleNamespace(OPTIMAL=2, INFEASIBLE=3, INF_OR_UNBD=4),
            quicksum=lambda terms: Budget(),
            gurobi=SimpleNamespace(version=lambda: (11, 0, 3)),
        )
        models = [
            fake_model([2, 10], objective=12),
            fake_model([3, 100]),
            fake_model([4, 70]),
            fake_model([], status=3),
            fake_model([2, 60]),
        ]
        network = small_network()
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model", side_effect=models) as build, \
                    patch("tdindp_capacities.generation._set_fairness_objective") as fairness:
                frontier = build_capacity_frontier(
                    network, directory, epsilon_floor=0.98, func_nonlinear=1
                )
            self.assertEqual(build.call_count, 5)
            for call in build.call_args_list:
                self.assertEqual(call.kwargs["func_nonlinear"], 1)
            for call in fairness.call_args_list:
                self.assertEqual(call.args[2], {"test": 10})
            for model, budget in zip(models[2:], (103, 74 * 0.99, 74 * 0.99**2)):
                self.assertAlmostEqual(model.addConstr.call_args.args[0], budget)
            self.assertEqual(frontier.n_points, 3)
            self.assertEqual(frontier.capacities(0)[("b", "a", "test", 0)], 10)
            self.assertEqual(frontier.objectives["sum_u"].iloc[0], 12)
            self.assertAlmostEqual(
                frontier.objectives["sum_log_u"].iloc[0], math.log(2) + math.log(10)
            )
            self.assertEqual(frontier.objectives["sum_u"].iloc[1], 74)
            self.assertAlmostEqual(
                frontier.objectives["sum_log_u"].iloc[1], math.log(4) + math.log(70)
            )
            check = frontier.objective_consistency()
            self.assertLess(check["sum_u_error"].abs().max(), 1e-9)
            self.assertLess(check["sum_log_u_error"].abs().max(), 1e-9)
            progress = next(Path(directory).rglob("progress.log")).read_text(encoding="utf-8")
            self.assertIn("Minimum-capacity anchor: optimize started", progress)
            self.assertIn("Fairness anchor: optimize finished", progress)
            self.assertIn("Sweep 2: optimize finished; status=3", progress)
            self.assertIn("continuing sweep", progress)
            self.assertIn("Run completed: points=3", progress)
            self.assertIn("effective FuncNonlinear=1; MIPGap=0.0001", progress)
            solver_logs = [call.kwargs["solver_log"] for call in build.call_args_list]
            self.assertEqual(len(set(solver_logs)), 5)

    def test_failed_anchor_does_not_create_output(self):
        solver = SimpleNamespace(
            GRB=SimpleNamespace(OPTIMAL=2),
            gurobi=SimpleNamespace(version=lambda: (11, 0, 3)),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model",
                          return_value=fake_model([], status=3)):
                with self.assertRaisesRegex(RuntimeError, "anchor failed"):
                    build_capacity_frontier(small_network(), directory)
            self.assertFalse((Path(directory) / "pareto_regenerated.csv").exists())
            progress = next(Path(directory).rglob("progress.log")).read_text(encoding="utf-8")
            self.assertIn("status=3", progress)
            self.assertIn("Run failed or interrupted", progress)
            self.assertIn("RuntimeError: Minimum-capacity anchor failed", progress)
            self.assertNotIn("Run completed", progress)

    def test_interrupts_leave_separate_logs_for_each_attempt(self):
        solver = SimpleNamespace(
            gurobi=SimpleNamespace(version=lambda: (11, 0, 3)),
        )
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "custom_logs"
            for attempt in range(2):
                model = fake_model([])
                model.optimize.side_effect = KeyboardInterrupt(f"interrupted attempt {attempt}")
                with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                        patch("tdindp_capacities.generation._capacity_model", return_value=model):
                    with self.assertRaises(KeyboardInterrupt):
                        build_capacity_frontier(small_network(), directory, log_dir=str(log_dir))
            logs = list(log_dir.rglob("progress.log"))
            self.assertEqual(len(logs), 2)
            contents = [path.read_text(encoding="utf-8") for path in logs]
            for attempt in range(2):
                self.assertEqual(
                    sum(f"KeyboardInterrupt: interrupted attempt {attempt}" in text for text in contents),
                    1,
                )
            self.assertFalse((Path(directory) / "pareto_regenerated.csv").exists())

    def test_progress_is_readable_while_a_solve_is_active(self):
        solver = SimpleNamespace(
            gurobi=SimpleNamespace(version=lambda: (11, 0, 3)),
        )
        with tempfile.TemporaryDirectory() as directory:
            def interrupted_solve():
                progress = next(Path(directory).rglob("progress.log")).read_text(encoding="utf-8")
                self.assertIn("Minimum-capacity anchor: optimize started", progress)
                self.assertNotIn("optimize finished", progress)
                raise KeyboardInterrupt("stop test solve")

            model = fake_model([])
            model.optimize.side_effect = interrupted_solve
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model", return_value=model):
                with self.assertRaises(KeyboardInterrupt):
                    build_capacity_frontier(small_network(), directory)

    def test_explicit_log_mode_rejects_unsupported_solver_version(self):
        solver = SimpleNamespace(gurobi=SimpleNamespace(version=lambda: (10, 0, 0)))
        with tempfile.TemporaryDirectory() as directory:
            with patch("tdindp_capacities.generation._gurobi", return_value=solver), \
                    patch("tdindp_capacities.generation._capacity_model") as model:
                with self.assertRaisesRegex(ValueError, "Gurobi 11"):
                    build_capacity_frontier(small_network(), directory, func_nonlinear=1)
                model.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is optional")
class FormulationTests(unittest.TestCase):
    def test_dependencies_use_actual_providers_and_preserve_coefficients(self):
        network = small_network()
        network.N_k["test"].append("c")
        network.b[("c", "test", "test", 0)] = 0
        network.gamma = {
            ("a", "b", "test", "test", 0): 2,
            ("c", "b", "test", "test", 0): 1,
        }
        with _capacity_model(network) as model:
            constraint = model.getConstrByName("A_7_('test', 'test', 'b', 0)")
            self.assertIsNotNone(constraint)
            self.assertEqual(constraint.Sense, ">")
            self.assertEqual(constraint.RHS, 0)
            row = model.getRow(constraint)
            coefficients = {
                row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())
            }
            self.assertEqual(
                coefficients,
                {"w_('a', 'test', 0)": 2, "w_('c', 'test', 0)": 1,
                 "w_('b', 'test', 0)": -1},
            )
            self.assertIsNone(model.getConstrByName("A_7_('test', 'test', 'a', 0)"))

    def test_cross_layer_dependency_blocks_service_when_all_providers_are_off(self):
        import gurobipy as gp

        network = small_network()
        network.A_k["test"] = [("a", "b"), ("b", "a")]
        network.K.insert(0, "provider")
        network.N_k["provider"] = ["p", "q", "r"]
        network.A_k["provider"] = [("p", "q"), ("q", "p"), ("q", "r"), ("r", "q")]
        network.L_k["provider"] = ["provider"]
        network.b.update({(i, "provider", "provider", 0): 0 for i in ("p", "q", "r")})
        network.c.update({
            (i, j, "provider", "provider", 0): 1
            for i, j in network.A_k["provider"]
        })
        network.gamma = {
            ("p", "b", "provider", "test", 0): 1,
            ("q", "b", "provider", "test", 0): 1,
            ("r", "a", "provider", "test", 1): 1,
        }
        with _capacity_model(network, func_nonlinear=1) as model:
            dependencies = [
                c for c in model.getConstrs() if c.ConstrName.startswith("A_7_")
            ]
            self.assertEqual(len(dependencies), 1)
            row = model.getRow(dependencies[0])
            self.assertEqual(
                {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())},
                {"w_('p', 'provider', 0)": 1, "w_('q', 'provider', 0)": 1,
                 "w_('b', 'test', 0)": -1},
            )
            model.getVarByName("w_('p', 'provider', 0)").UB = 0
            other_provider = model.getVarByName("w_('q', 'provider', 0)")
            other_provider.UB = 0
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.INFEASIBLE)
            other_provider.LB = other_provider.UB = 1
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.OPTIMAL)

    def test_real_solver_writes_progress_and_parameter_files(self):
        import gurobipy as gp

        network = small_network()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "solver.log"
            with _capacity_model(network, solver_log=str(log_path), func_nonlinear=1) as model:
                self.assertEqual(model.Params.LogToConsole, 0)
                self.assertEqual(model.Params.FuncNonlinear, 1)
                self.assertEqual(model.Params.MIPGap, model.getParamInfo("MIPGap")[-1])
                self.assertEqual(
                    model.Params.FeasibilityTol, model.getParamInfo("FeasibilityTol")[-1]
                )
                _optimize_logged(model, "Test model", logging.getLogger(__name__), str(log_path))
                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("Gurobi", text)
            self.assertIn("Optimal solution found", text)
            self.assertTrue(log_path.with_suffix(".prm").is_file())
            self.assertIn("FuncNonlinear", log_path.with_suffix(".prm").read_text())

    def test_explicit_piecewise_linear_mode_does_not_change_gap(self):
        with _capacity_model(small_network(), func_nonlinear=0) as model:
            self.assertEqual(model.Params.FuncNonlinear, 0)
            self.assertEqual(model.Params.MIPGap, model.getParamInfo("MIPGap")[-1])

    def test_model_variables_constraints_and_defaults(self):
        import gurobipy as gp

        network = small_network()
        with _capacity_model(network) as model:
            variables = model.getVars()
            self.assertEqual(model.NumBinVars, 4)
            self.assertEqual(model.NumGenConstrs, 2)
            self.assertEqual(model.NumQConstrs, 6)
            capacity_name = "u_('a', 'b', 'test', 0)"
            capacity_vars = [v for v in variables if v.VarName == capacity_name]
            self.assertEqual(len(capacity_vars), 1)
            self.assertEqual(capacity_vars[0].Obj, 1)
            self.assertEqual(capacity_vars[0].LB, 0)
            self.assertGreaterEqual(capacity_vars[0].UB, gp.GRB.INFINITY)
            self.assertEqual(model.Params.FuncNonlinear, model.getParamInfo("FuncNonlinear")[-1])
            self.assertEqual(model.Params.FeasibilityTol, model.getParamInfo("FeasibilityTol")[-1])
            self.assertIsNotNone(model.getConstrByName("meet_demand_('b', 'test', 'test', 0)"))
            excess = model.getVarByName("delta_plus_('a', 'test', 'test', 0)")
            self.assertGreaterEqual(excess.UB, gp.GRB.INFINITY)
            self.assertEqual(excess.Obj, 0)

            _set_fairness_objective(model, network, {"test": 10})
            log_variable = model.getVarByName("log_u_('a', 'b', 'test', 0)")
            self.assertEqual(log_variable.Obj, 1)
            self.assertGreaterEqual(log_variable.UB, gp.GRB.INFINITY)
            bounds = [c for c in model.getConstrs() if c.ConstrName.startswith("max_cap_")]
            self.assertEqual(len(bounds), 2)
            self.assertTrue(all(abs(c.RHS - math.log(10)) < 1e-12 for c in bounds))


if __name__ == "__main__":
    unittest.main()
