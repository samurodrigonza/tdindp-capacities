"""Epsilon sweep whose log-capacity ceilings follow the previous solve.

This is an experimental variant of ``generation.build_capacity_frontier``, which
stays the default. There, each layer's ceiling is set once from the minimum
solve. Here, the minimum solve, the fairness anchor, and the first budgeted
solve are unchanged, and each later solve is capped at the layer maxima of the
last feasible solve.

No solution can exceed its ceiling, so the ceilings can only fall.
"""

from __future__ import annotations

import os
import platform
from time import perf_counter
from typing import Optional

import pandas as pd

from .capacities import CapacityFrontier, load_frontier
from .generation import (
    _capacity_model, _capacity_values, _gurobi, _optimize_logged, _run_logging,
    _set_fairness_objective, _validate_request, _write_frontier,
)
from .networks import DATA_DIR, Networks


def _layer_maxima(capacities: dict, networks: Networks) -> dict:
    return {
        k: max(capacities[(i, j, k, 0)] for i, j in networks.A_k[k])
        for k in networks.K
    }


def build_moving_ceiling_frontier(
    networks: Networks,
    data_dir: str = DATA_DIR,
    filename: str = "pareto_moving_ceiling.csv",
    epsilon: float = 1.0,
    epsilon_decay: float = 0.99,
    epsilon_floor: float = 0.1,
    log_dir: Optional[str] = None,
    func_nonlinear: Optional[int] = None,
) -> CapacityFrontier:
    """Run the sweep with moving layer ceilings, saving to new CSV files.

    The frontier has the usual CSV format. A ``<name>_trace.csv`` beside it
    records every budgeted solve: its ceilings, budget, and outcome. Existing
    files are never overwritten.
    """
    path = os.path.join(data_dir, filename)
    trace_path = os.path.splitext(path)[0] + "_trace.csv"
    for existing in (path, trace_path):
        if os.path.exists(existing):
            raise FileExistsError(f"Refusing to overwrite existing file: {existing}")
    _validate_request(networks, epsilon, epsilon_decay, epsilon_floor, func_nonlinear)
    with _run_logging(path, log_dir) as (logger, run_dir):
        return _generate(
            networks, data_dir, filename, trace_path, epsilon, epsilon_decay,
            epsilon_floor, logger, run_dir, func_nonlinear,
        )


def _generate(
    networks, data_dir, filename, trace_path, epsilon, epsilon_decay, epsilon_floor,
    logger, run_dir, func_nonlinear,
) -> CapacityFrontier:
    path = os.path.join(data_dir, filename)
    gp = _gurobi()
    if func_nonlinear is not None and gp.gurobi.version()[0] < 11:
        raise ValueError("Explicit func_nonlinear selection requires Gurobi 11 or newer")
    start = perf_counter()
    logger.info(
        "Environment: Python=%s; Gurobi=%s; platform=%s; logical_cpus=%s",
        platform.python_version(), gp.gurobi.version(), platform.platform(), os.cpu_count(),
    )
    logger.info(
        "Moving ceilings; epsilon=%g; epsilon_decay=%g; epsilon_floor=%g; FuncNonlinear=%s",
        epsilon, epsilon_decay, epsilon_floor,
        "unchanged" if func_nonlinear is None else func_nonlinear,
    )

    solver_log = os.path.join(run_dir, "minimum-anchor.log")
    with _capacity_model(
        networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
    ) as model:
        _optimize_logged(model, "Minimum-capacity anchor", logger, solver_log)
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Minimum-capacity anchor failed: status {model.Status}")
        optimal_total = model.ObjVal
        minimum = _capacity_values(model)
    ceilings = _layer_maxima(minimum, networks)
    logger.info("Minimum objective=%.12g; initial ceilings=%s", optimal_total, ceilings)

    solver_log = os.path.join(run_dir, "fairness-anchor.log")
    with _capacity_model(
        networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
    ) as model:
        _set_fairness_objective(model, networks, ceilings)
        _optimize_logged(model, "Fairness anchor", logger, solver_log)
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Fairness anchor failed: status {model.Status}")
        actual_total = sum(_capacity_values(model).values())

    solutions = [minimum]
    trace = []
    iteration = 0
    while actual_total > optimal_total and epsilon > epsilon_floor:
        iteration += 1
        budget = actual_total * epsilon
        solver_log = os.path.join(run_dir, f"sweep-{iteration:04d}.log")
        logger.info(
            "Sweep %d: epsilon=%.12g; budget=%.12g; ceilings=%s; retained_points=%d",
            iteration, epsilon, budget, ceilings, len(solutions),
        )
        record = {"iteration": iteration, "epsilon": epsilon, "budget": budget}
        record.update({"ceiling_" + k: ceilings[k] for k in networks.K})
        with _capacity_model(
            networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
        ) as model:
            _set_fairness_objective(model, networks, ceilings)
            model.addConstr(
                gp.quicksum(
                    model.getVarByName("u_" + str((i, j, k, t)))
                    for k in networks.K for i, j in networks.A_k[k] for t in networks.T
                ) <= budget
            )
            model.update()
            _optimize_logged(model, f"Sweep {iteration}", logger, solver_log)
            record["status"] = model.Status
            if model.Status == gp.GRB.OPTIMAL:
                actual = _capacity_values(model)
                actual_total = sum(actual.values())
                record["sum_u"] = actual_total
                if actual not in solutions:
                    solutions.append(actual)
                    record["outcome"] = "accepted"
                else:
                    record["outcome"] = "duplicate"
                ceilings = _layer_maxima(actual, networks)
                logger.info(
                    "Sweep %d: %s; sum_u=%.12g; next ceilings=%s",
                    iteration, record["outcome"], actual_total, ceilings,
                )
            elif model.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                record["outcome"] = "infeasible"
                logger.warning(
                    "No solution at epsilon=%g, budget=%g (status=%s); continuing sweep",
                    epsilon, budget, model.Status,
                )
            else:
                raise RuntimeError(f"Capacity sweep failed: status {model.Status}")
        trace.append(record)
        epsilon *= epsilon_decay

    logger.info(
        "Sweep ended: iterations=%d; epsilon=%.12g; last_total=%.12g; minimum_objective=%.12g",
        iteration, epsilon, actual_total, optimal_total,
    )
    _write_frontier(path, solutions)
    with open(trace_path, "x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(trace).to_csv(stream, index=False)
    frontier = load_frontier(data_dir=data_dir, filename=filename)
    logger.info(
        "Run completed: points=%d; capacity_columns=%d; elapsed=%.3fs; output=%s",
        frontier.n_points, frontier.n_arcs, perf_counter() - start, os.path.abspath(path),
    )
    return frontier
