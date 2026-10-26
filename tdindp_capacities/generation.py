"""The nominal build branch of stochasticFairness's find_capacities routine.

This extraction retains the source's formulation, Gurobi defaults, duplicate
endpoint behavior, and objective bookkeeping. It uses the source's shipped
configuration: T = [0], no damaged components, and no critical-demand nodes.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import platform
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import pandas as pd

from .capacities import CapacityFrontier, load_frontier
from .networks import DATA_DIR, Networks


def _gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError(
            "Capacity generation requires gurobipy and a valid Gurobi license. "
            "Install the optional dependency with pip install -e '.[gurobi]'."
        ) from exc
    return gp


@contextmanager
def _run_logging(output_path: str, log_dir: Optional[str]):
    if log_dir is None:
        log_dir = os.path.splitext(os.path.abspath(output_path))[0] + "_logs"
    os.makedirs(log_dir, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    run_dir = tempfile.mkdtemp(prefix=prefix, dir=os.path.abspath(log_dir))
    logger = logging.Logger(__name__, level=logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(run_dir, "progress.log"), encoding="utf-8"
    )
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    start = perf_counter()
    completed = False
    try:
        logger.info("Run started; output=%s; logs=%s", os.path.abspath(output_path), run_dir)
        yield logger, run_dir
        completed = True
    finally:
        if not completed:
            logger.error(
                "Run failed or interrupted after %.3fs; no successful completion recorded",
                perf_counter() - start, exc_info=True,
            )
        for handler in (file_handler, console_handler):
            logger.removeHandler(handler)
            handler.close()


def _optimize_logged(model, stage: str, logger, solver_log: str) -> None:
    model.write(os.path.splitext(solver_log)[0] + ".prm")
    log_mode = (
        model.Params.FuncNonlinear
        if _gurobi().gurobi.version()[0] >= 11 else "unavailable before Gurobi 11"
    )
    logger.info(
        "%s: effective FuncNonlinear=%s; MIPGap=%s",
        stage, log_mode, model.Params.MIPGap,
    )
    logger.info("%s: optimize started; solver_log=%s", stage, solver_log)
    start = perf_counter()
    model.optimize()
    logger.info(
        "%s: optimize finished; status=%s; elapsed=%.3fs; solutions=%s",
        stage, model.Status, perf_counter() - start, model.SolCount,
    )
    if model.SolCount > 0:
        logger.info("%s: objective=%.12g", stage, model.ObjVal)


def _capacity_model(
    networks: Networks,
    solver_log: Optional[str] = None,
    func_nonlinear: Optional[int] = None,
):
    """Build tdINDP_free_cap for the original nominal, single-period inputs."""
    gp = _gurobi()
    K, T = networks.K, networks.T
    N_k, A_k, L_k = networks.N_k, networks.A_k, networks.L_k
    b, gamma = networks.b, networks.gamma
    model = gp.Model("tdINDP")
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    if solver_log is not None:
        model.setParam("LogFile", solver_log)
        model.setParam("LogToConsole", 0)
        model.setParam("OutputFlag", 1)
    model.setParam("InfUnbdInfo", 1)
    if func_nonlinear is not None:
        model.setParam("FuncNonlinear", func_nonlinear)

    demand = {k: [i for i in N_k[k] if b[(i, k, k, 0)] < 0] for k in K}
    supply = {k: [i for i in N_k[k] if b[(i, k, k, 0)] > 0] for k in K}
    total_cost = model.addVar(
        vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="total_cost"
    )
    delta_plus = {
        (i, k, l, t): model.addVar(
            vtype=gp.GRB.CONTINUOUS, name="delta_plus_" + str((i, k, l, t))
        )
        for k in K for i in supply[k] for l in L_k[k] for t in T
    }
    delta_minus = {
        (i, k, l, t): model.addVar(
            vtype=gp.GRB.CONTINUOUS, name="delta_minus_" + str((i, k, l, t))
        )
        for k in K for i in demand[k] for l in L_k[k] for t in T
    }
    # Do not deduplicate A_k: repeated keys create variables before overwriting
    # dictionary entries, just as in the source.
    x = {
        (i, j, k, l, t): model.addVar(
            vtype=gp.GRB.CONTINUOUS, name="x_" + str((i, j, k, l, t))
        )
        for k in K for i, j in A_k[k] for l in L_k[k] for t in T
    }
    u = {
        (i, j, k, t): model.addVar(
            vtype=gp.GRB.CONTINUOUS, name="u_" + str((i, j, k, t))
        )
        for k in K for i, j in A_k[k] for t in T
    }
    log_u = {
        (i, j, k, t): model.addVar(
            vtype=gp.GRB.CONTINUOUS, name="log_u_" + str((i, j, k, t))
        )
        for k in K for i, j in A_k[k] for t in T
    }
    w = {
        (i, k, t): model.addVar(vtype=gp.GRB.BINARY, name="w_" + str((i, k, t)))
        for k in K for i in N_k[k] for t in T
    }
    y = {
        (i, j, k, t): model.addVar(
            vtype=gp.GRB.BINARY, name="y_" + str((i, j, k, t))
        )
        for k in K for i, j in A_k[k] for t in T
    }
    cost_4 = gp.quicksum(
        networks.M_plus[(i, k, l, t)] * delta_plus[(i, k, l, t)]
        for k in K for l in L_k[k] for i in supply[k] for t in T
    ) + gp.quicksum(
        networks.M_minus[(i, k, l, t)] * delta_minus[(i, k, l, t)]
        for k in K for l in L_k[k] for i in demand[k] for t in T
    )
    cost_5 = gp.quicksum(
        networks.c[(i, j, k, l, t)] * x[(i, j, k, l, t)]
        for k in K for l in L_k[k] for i, j in A_k[k] for t in T
    )
    model.addConstr(total_cost == cost_4 + cost_5, name="total_cost_constraint")

    for k in K:
        for i in N_k[k]:
            for l in L_k[k]:
                for t in T:
                    outgoing = gp.quicksum(
                        x[(i, j, k, l, t)] for j in N_k[k] if (i, j) in A_k[k]
                    )
                    incoming = gp.quicksum(
                        x[(j, i, k, l, t)] for j in N_k[k] if (j, i) in A_k[k]
                    )
                    rhs = b[(i, k, l, t)]
                    if i in demand[k]:
                        rhs += delta_minus[(i, k, l, t)]
                    elif i in supply[k]:
                        rhs -= delta_plus[(i, k, l, t)]
                    model.addConstr(
                        outgoing - incoming == rhs, name="A_2_" + str((i, k, l, t))
                    )

    for k in K:
        for i, j in A_k[k]:
            for t in T:
                flow = gp.quicksum(x[(i, j, k, l, t)] for l in L_k[k])
                model.addConstr(
                    flow <= u[(i, j, k, t)] * y[(i, j, k, t)],
                    name="A_3_" + str((k, i, j, t)),
                )
                model.addConstr(
                    flow <= u[(i, j, k, t)] * w[(i, k, t)],
                    name="A_4_" + str((k, i, j, t)),
                )
                model.addConstr(
                    flow <= u[(i, j, k, t)] * w[(j, k, t)],
                    name="A_5_" + str((k, i, j, t)),
                )

    # The source guard uses i left over from the preceding arc loop.
    # Retain it rather than silently changing which dependencies are imposed.
    for k in K:
        for k_tilde in K:
            for j in N_k[k_tilde]:
                for t in T:
                    if (i, j, k, k_tilde, t) in gamma:
                        model.addConstr(
                            gp.quicksum(
                                w[(i, k, t)] * gamma[(i, j, k, k_tilde, t)]
                                for i in N_k[k]
                            ) >= w[(j, k_tilde, t)],
                            name="A_7_" + str((k, k_tilde, j, t)),
                        )
    model.update()
    for k in K:
        for i in demand[k]:
            for l in L_k[k]:
                for t in T:
                    model.addConstr(
                        delta_minus[(i, k, l, t)] == 0,
                        name="meet_demand_" + str((i, k, l, t)),
                    )
    for k in K:
        for t in T:
            for i, j in A_k[k]:
                model.addGenConstrLog(
                    u[(i, j, k, t)], log_u[(i, j, k, t)],
                    name="log_u_constr_" + str((i, j, k, t)),
                )
                if (j, i) in A_k[k]:
                    model.addConstr(
                        u[(i, j, k, t)] == u[(j, i, k, t)],
                        name="bidir_cap_" + str((i, j, k, t)),
                    )
    model.setObjective(
        gp.quicksum(u[(i, j, k, t)] for k in K for i, j in A_k[k] for t in T),
        gp.GRB.MINIMIZE,
    )
    model.update()
    return model


def _capacity_values(model) -> dict:
    return {
        ast.literal_eval(variable.VarName[2:]): variable.X
        for variable in model.getVars() if variable.VarName[0] == "u"
    }


def _set_fairness_objective(model, networks: Networks, upper_bounds: dict) -> None:
    gp = _gurobi()
    model.setObjective(
        gp.quicksum(
            model.getVarByName("log_u_" + str((i, j, k, t)))
            for k in networks.K for i, j in networks.A_k[k] for t in networks.T
        ),
        gp.GRB.MAXIMIZE,
    )
    model.update()
    for k in networks.K:
        for i, j in networks.A_k[k]:
            for t in networks.T:
                model.addConstr(
                    model.getVarByName("log_u_" + str((i, j, k, t)))
                    <= math.log(upper_bounds[k]),
                    name="max_cap_" + str((i, j, k, t)),
                )
                model.update()


def build_capacity_frontier(
    networks: Networks,
    data_dir: str = DATA_DIR,
    filename: str = "pareto_regenerated.csv",
    epsilon: float = 1.0,
    epsilon_decay: float = 0.99,
    epsilon_floor: float = 0.1,
    log_dir: Optional[str] = None,
    func_nonlinear: Optional[int] = None,
) -> CapacityFrontier:
    """Run the source's single-period nominal sweep, saving to a new CSV.

    Defaults retain the original epsilon schedule. The historical frontier
    and the reader's default selection are not modified. Each invocation gets
    a unique log directory under log_dir, or beside the output CSV by default.
    By default only logging settings change. With Gurobi 11+, explicitly pass
    func_nonlinear=1 for nonlinear logs or 0 for static piecewise-linear logs.
    None leaves the solver's log-handling setting untouched. MIPGap is never
    overridden.
    """
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing frontier: {path}")
    if networks.T != [0]:
        raise ValueError("This extraction supports the source's nominal T = [0] only")
    if not networks.K or any(
        not networks.N_k[k] or not networks.A_k[k] for k in networks.K
    ):
        raise ValueError("Capacity generation requires nonempty layers, nodes, and arcs")
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    if not 0 < epsilon_decay < 1:
        raise ValueError("epsilon_decay must be in (0, 1)")
    if not 0 < epsilon_floor < epsilon:
        raise ValueError("epsilon_floor must be in (0, epsilon)")
    if func_nonlinear is not None and func_nonlinear not in (0, 1):
        raise ValueError("func_nonlinear must be None, 0, or 1")
    with _run_logging(path, log_dir) as (logger, run_dir):
        return _generate_frontier(
            networks, data_dir, filename, epsilon, epsilon_decay, epsilon_floor,
            logger, run_dir, func_nonlinear,
        )


def _generate_frontier(
    networks, data_dir, filename, epsilon, epsilon_decay, epsilon_floor, logger, run_dir,
    func_nonlinear,
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
        "Schedule: epsilon=%g; epsilon_decay=%g; epsilon_floor=%g; layers=%s; periods=%s",
        epsilon, epsilon_decay, epsilon_floor, networks.K, networks.T,
    )
    logger.info(
        "Requested FuncNonlinear=%s; MIPGap is not overridden",
        "unchanged" if func_nonlinear is None else func_nonlinear,
    )

    solver_log = os.path.join(run_dir, "minimum-anchor.log")
    logger.info("Minimum-capacity anchor: building model")
    with _capacity_model(
        networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
    ) as model:
        _optimize_logged(model, "Minimum-capacity anchor", logger, solver_log)
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Minimum-capacity anchor failed: status {model.Status}")
        optimal_total = model.ObjVal
        minimum = _capacity_values(model)
    upper_bounds = {
        k: max(minimum[(i, j, k, 0)] for i, j in networks.A_k[k])
        for k in networks.K
    }
    logger.info("Minimum objective=%.12g; layer upper bounds=%s", optimal_total, upper_bounds)

    solver_log = os.path.join(run_dir, "fairness-anchor.log")
    logger.info("Fairness anchor: building model")
    with _capacity_model(
        networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
    ) as model:
        _set_fairness_objective(model, networks, upper_bounds)
        _optimize_logged(model, "Fairness anchor", logger, solver_log)
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Fairness anchor failed: status {model.Status}")
        actual_total = sum(_capacity_values(model).values())

    minimum["sum_log_u"] = sum(math.log(value) for value in minimum.values())
    minimum["sum_u"] = optimal_total
    solutions = [minimum]
    iteration = 0
    while actual_total > optimal_total and epsilon > epsilon_floor:
        iteration += 1
        solver_log = os.path.join(run_dir, f"sweep-{iteration:04d}.log")
        logger.info(
            "Sweep %d: building model; epsilon=%.12g; budget=%.12g; retained_points=%d",
            iteration, epsilon, actual_total * epsilon, len(solutions),
        )
        with _capacity_model(
            networks, solver_log=solver_log, func_nonlinear=func_nonlinear,
        ) as model:
            _set_fairness_objective(model, networks, upper_bounds)
            model.addConstr(
                gp.quicksum(
                    model.getVarByName("u_" + str((i, j, k, t)))
                    for k in networks.K for i, j in networks.A_k[k] for t in networks.T
                ) <= actual_total * epsilon
            )
            model.update()
            _optimize_logged(model, f"Sweep {iteration}", logger, solver_log)
            if model.Status == gp.GRB.OPTIMAL:
                actual = _capacity_values(model)
                actual_total = sum(
                    variable.X for variable in model.getVars() if variable.VarName[0] == "u"
                )
                actual["sum_log_u"] = sum(
                    math.log(variable.X + 1e-6)
                    for variable in model.getVars() if variable.VarName[0] == "u"
                )
                actual["sum_u"] = actual_total
                if actual not in solutions:
                    solutions.append(actual)
                    logger.info(
                        "Sweep %d: accepted point; sum_u=%.12g; sum_log_u=%.12g; retained_points=%d",
                        iteration, actual_total, actual["sum_log_u"], len(solutions),
                    )
                else:
                    logger.info("Sweep %d: duplicate point; not appended", iteration)
            elif model.Status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                logger.warning(
                    "No solution at epsilon=%g, budget=%g (status=%s); continuing sweep",
                    epsilon, actual_total * epsilon, model.Status,
                )
            else:
                raise RuntimeError(f"Capacity sweep failed: status {model.Status}")
        epsilon *= epsilon_decay

    logger.info(
        "Sweep ended: iterations=%d; epsilon=%.12g; last_total=%.12g; minimum_objective=%.12g",
        iteration, epsilon, actual_total, optimal_total,
    )
    data = {key: [float(solution[key]) for solution in solutions] for key in solutions[0]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    logger.info("Writing %d points to %s", len(solutions), os.path.abspath(path))
    with open(path, "x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(data, columns=data.keys()).to_csv(stream)
    frontier = load_frontier(data_dir=data_dir, filename=filename)
    logger.info(
        "Run completed: points=%d; capacity_columns=%d; elapsed=%.3fs; output=%s",
        frontier.n_points, frontier.n_arcs, perf_counter() - start, os.path.abspath(path),
    )
    return frontier
