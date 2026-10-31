"""Solves and bounds behind the paper's ceiling-rule analysis.

``layer_throughput_bounds`` needs no solver. Everything else needs Gurobi and
uses the same nominal model as the generator, so its results can be compared
with the frontiers directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .generation import (
    _capacity_model, _capacity_values, _gurobi, _set_fairness_objective,
)
from .networks import Networks


def undirected_connections(networks: Networks) -> Dict[str, List[Tuple[str, str]]]:
    """Each layer's connections, once per reverse pair, in first-seen order."""
    connections = {}
    for k in networks.K:
        seen = set()
        connections[k] = []
        for i, j in networks.A_k[k]:
            if frozenset((i, j)) not in seen:
                seen.add(frozenset((i, j)))
                connections[k].append((i, j))
    return connections


def layer_throughput_bounds(networks: Networks) -> Dict[str, dict]:
    """A lower bound on each layer's largest capacity, from one node's balance.

    A node with net availability ``b`` and ``d`` neighbours must push ``|b|``
    through ``d`` arcs, so one of them carries at least ``|b| / d``. Returns the
    tightest such bound per layer, with the node and values that produce it.
    """
    bounds = {}
    for k in networks.K:
        best = None
        for i in networks.N_k[k]:
            degree = len({head for tail, head in networks.A_k[k] if tail == i})
            if degree:
                availability = abs(networks.b[(i, k, k, 0)])
                if best is None or availability / degree > best["bound"]:
                    best = {"bound": availability / degree, "node": i,
                            "availability": availability, "degree": degree}
        bounds[k] = best
    return bounds


def layer_bottlenecks(networks: Networks) -> Dict[str, float]:
    """The smallest largest-arc flow that can serve each layer's demand.

    A linear program per layer, with unmet demand fixed at zero and excess
    supply allowed, as in the capacity model. No ceiling below this value can
    admit nominal service.
    """
    gp = _gurobi()
    result = {}
    for k in networks.K:
        arcs = list(dict.fromkeys(networks.A_k[k]))
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            flow = {arc: model.addVar() for arc in arcs}
            level = model.addVar()
            for arc in arcs:
                model.addConstr(flow[arc] <= level)
            for i in networks.N_k[k]:
                outgoing = gp.quicksum(flow[a] for a in arcs if a[0] == i)
                incoming = gp.quicksum(flow[a] for a in arcs if a[1] == i)
                availability = networks.b[(i, k, k, 0)]
                if availability > 0:
                    model.addConstr(outgoing - incoming <= availability)
                else:
                    model.addConstr(outgoing - incoming == availability)
            model.setObjective(level, gp.GRB.MINIMIZE)
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                raise RuntimeError(f"Bottleneck LP failed for {k}: status {model.Status}")
            result[k] = model.ObjVal
    return result


def net_flow_ranges(
    networks: Networks,
    capacities: dict,
    k: str,
    connections: List[Tuple[str, str]],
    slack: float = 1e-6,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """The smallest and largest net flow on connections under fixed capacities.

    Each result bounds ``x_ij - x_ji`` over every nominal flow of layer ``k`` that
    fits the capacities. ``slack`` widens them by the solver's feasibility tolerance,
    since exported capacities can sit exactly at the flow they carry.
    """
    gp = _gurobi()
    arcs = list(dict.fromkeys(networks.A_k[k]))
    result = {}
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        flow = {arc: model.addVar(ub=capacities[arc + (k, 0)] + slack) for arc in arcs}
        for i in networks.N_k[k]:
            outgoing = gp.quicksum(flow[a] for a in arcs if a[0] == i)
            incoming = gp.quicksum(flow[a] for a in arcs if a[1] == i)
            availability = networks.b[(i, k, k, 0)]
            if availability > 0:
                model.addConstr(outgoing - incoming <= availability)
            else:
                model.addConstr(outgoing - incoming == availability)
        for i, j in connections:
            net = flow[(i, j)] - flow[(j, i)]
            bounds = []
            for sense in (gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE):
                model.setObjective(net, sense)
                model.optimize()
                if model.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f"Flow range failed on {(i, j)}: status {model.Status}")
                bounds.append(model.ObjVal)
            result[(i, j)] = tuple(bounds)
    return result


def solve_under_ceilings(
    networks: Networks,
    ceilings: dict,
    budget: Optional[float] = None,
    minimize_total: bool = False,
    func_nonlinear: Optional[int] = None,
    all_functional: bool = False,
) -> dict:
    """One solve of the capacity model under given layer ceilings.

    Maximizes the log-capacity sum, or minimizes total capacity when
    ``minimize_total`` is set, optionally under a total-capacity budget. With
    ``all_functional``, every functionality binary is fixed at one. Returns the
    solver status, objective and bound, and the capacities when optimal.
    """
    gp = _gurobi()
    with _capacity_model(networks, func_nonlinear=func_nonlinear) as model:
        _set_fairness_objective(model, networks, ceilings)
        if all_functional:
            for variable in model.getVars():
                if variable.VarName.startswith(("w_", "y_")):
                    variable.LB = 1
        total = gp.quicksum(
            model.getVarByName("u_" + str((i, j, k, t)))
            for k in networks.K for i, j in networks.A_k[k] for t in networks.T
        )
        if budget is not None:
            model.addConstr(total <= budget)
        if minimize_total:
            model.setObjective(total, gp.GRB.MINIMIZE)
        model.optimize()
        result = {"status": model.Status, "capacities": None,
                  "objective": None, "bound": None}
        if model.Status == gp.GRB.OPTIMAL:
            result.update(
                capacities=_capacity_values(model),
                objective=model.ObjVal,
                bound=model.ObjBound,
            )
        return result


def recheck_capacities(
    networks: Networks,
    capacities: dict,
    allowance: float = 0.0,
    func_nonlinear: Optional[int] = None,
) -> bool:
    """Whether a capacity vector passes the all-functional nominal recheck.

    Fixes every capacity, within ``allowance``, and every functionality binary
    at one in the full model, leaving dependency, reverse-capacity, and logarithm
    constraints in place at the solver's default feasibility tolerance.
    """
    gp = _gurobi()
    with _capacity_model(networks, func_nonlinear=func_nonlinear) as model:
        for key, value in capacities.items():
            variable = model.getVarByName("u_" + str(key))
            variable.LB, variable.UB = value - allowance, value + allowance
        for variable in model.getVars():
            if variable.VarName.startswith(("w_", "y_")):
                variable.LB = 1
        model.setObjective(0)
        model.optimize()
        return model.Status == gp.GRB.OPTIMAL
