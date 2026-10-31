"""Compute the paper's ceiling-rule evidence from the frontier files.

Run from the repository root, after the two sweeps exist in ``data/``::

    python scripts/run_paper_analysis.py

Requires Gurobi. Writes the matched-budget solutions, the coarser-schedule runs,
and ``paper_analysis_2026-10-31.json`` to ``data/``; nothing is overwritten.
"""

import json
import math
import os
from time import perf_counter

import pandas as pd

from tdindp_capacities import (
    DATA_DIR, PRE_FIX_DATA_DIR, build_moving_ceiling_frontier, load_benchmark,
    load_frontier, load_networks,
)
from tdindp_capacities.analysis import (
    layer_bottlenecks, layer_throughput_bounds, net_flow_ranges, recheck_capacities,
    solve_under_ceilings, undirected_connections,
)

TAG = "2026-10-31"
MOVING = f"pareto_moving_ceiling_{TAG}.csv"
REPEATED_ARC = f"pareto_regenerated_repeated_arc_{TAG}.csv"
#: Coarser budget schedules for the moving rule; the main runs use 0.99.
DECAYS = (0.98, 0.95, 0.9)
ALLOWANCES = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
FUNC_NONLINEAR = 1
#: Budgets inside the range that the moving sweep's schedule stepped over.
GRID = (10900.0, 11000.0, 11250.0, 11750.0, 12000.0, 12500.0)
#: Gas connections around node 1, oriented along the route the displaced flow takes.
GAS_ROUTE = (
    ("Gas_node_1", "Gas_node_5"), ("Gas_node_1", "Gas_node_13"), ("Gas_node_13", "Gas_node_6"),
    ("Gas_node_6", "Gas_node_11"), ("Gas_node_11", "Gas_node_2"), ("Gas_node_2", "Gas_node_10"),
    ("Gas_node_10", "Gas_node_4"), ("Gas_node_5", "Gas_node_10"), ("Gas_node_5", "Gas_node_4"),
)
#: Net flow below this is treated as zero when classifying connections.
FLOW_TOLERANCE = 1e-4


def write_new_csv(frame, filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream)


def log_sum(capacities):
    return sum(math.log(value) for value in capacities.values())


def layer_maxima(capacities, layers):
    return {
        k: max(value for key, value in capacities.items() if key[2] == k)
        for k in layers
    }


def recheck_failure(networks, capacities):
    """``None`` if the vector passes the recheck, else the size of the failure."""
    if recheck_capacities(networks, capacities, func_nonlinear=FUNC_NONLINEAR):
        return None
    mismatch = max(
        abs(value - capacities[(j, i, k, t)])
        for (i, j, k, t), value in capacities.items()
    )
    allowance = next(
        (a for a in ALLOWANCES
         if recheck_capacities(networks, capacities, a, FUNC_NONLINEAR)),
        None,
    )
    return {"reverse_mismatch": mismatch, "allowance": allowance}


def recheck_frontier(networks, frontier):
    failures = []
    for row in range(frontier.n_points):
        failure = recheck_failure(networks, frontier.capacities(row))
        if failure:
            failures.append(dict(failure, row=row))
    return {"points": frontier.n_points, "passed": frontier.n_points - len(failures),
            "failures": failures}


def layer_totals(capacities, layers):
    return {k: sum(v for key, v in capacities.items() if key[2] == k) for k in layers}


def reference_minimum_flows(networks, capacities):
    """How the connections at the reference's smallest capacity are actually used."""
    summary = {"connections": 0, "forced_flow": 0, "saturated": 0, "can_be_idle": []}
    connections = undirected_connections(networks)
    for k in networks.K:
        chosen = [(i, j) for i, j in connections[k] if capacities[(i, j, k, 0)] < 1 + 1e-3]
        ranges = net_flow_ranges(networks, capacities, k, chosen)
        for (i, j), (low, high) in ranges.items():
            summary["connections"] += 1
            if low > FLOW_TOLERANCE or high < -FLOW_TOLERANCE:
                summary["forced_flow"] += 1
                if min(abs(low), abs(high)) > capacities[(i, j, k, 0)] - FLOW_TOLERANCE:
                    summary["saturated"] += 1
            else:
                summary["can_be_idle"].append(f"{i}-{j}")
    return summary


def gas_route_flows(networks, fixed_capacities, moving_capacities):
    rows = {}
    fixed_ranges = net_flow_ranges(networks, fixed_capacities, "Gas", list(GAS_ROUTE))
    moving_ranges = net_flow_ranges(networks, moving_capacities, "Gas", list(GAS_ROUTE))
    for connection in GAS_ROUTE:
        i, j = connection
        rows[f"{i}-{j}"] = {
            "fixed_capacity": fixed_capacities[(i, j, "Gas", 0)],
            "moving_capacity": moving_capacities[(i, j, "Gas", 0)],
            "fixed_flow": fixed_ranges[connection],
            "moving_flow": moving_ranges[connection],
        }
    return rows


def single_ceiling_effects(networks, base_ceilings, bounds, budget):
    """Tighten one layer's ceiling at a time, to separate the layers' effects."""
    summaries = {}
    for k in networks.K:
        ceilings = dict(base_ceilings)
        ceilings[k] = bounds[k]["bound"]
        solve = solve_under_ceilings(networks, ceilings, budget=budget, func_nonlinear=FUNC_NONLINEAR)
        summaries[k] = {
            "status": solve["status"],
            "log_sum": log_sum(solve["capacities"]),
            "minimum": min(solve["capacities"].values()),
            "layer_totals": layer_totals(solve["capacities"], networks.K),
        }
    return summaries


def decay_sensitivity(networks):
    """Rerun the moving rule on coarser schedules, keeping the outputs apart."""
    directory = os.path.join(DATA_DIR, f"decay_sensitivity_{TAG}")
    log_dir = os.path.join(DATA_DIR, f"decay_sensitivity_{TAG}_logs")
    os.makedirs(directory, exist_ok=True)
    summaries = []
    for decay in DECAYS:
        print(f"Moving ceilings with decay {decay}", flush=True)
        solve_start = perf_counter()
        frontier = build_moving_ceiling_frontier(
            networks, data_dir=directory, filename=f"moving_decay_{decay}.csv",
            epsilon_decay=decay, log_dir=log_dir, func_nonlinear=FUNC_NONLINEAR,
        )
        last = frontier.capacities(-1)
        ceilings = layer_maxima(last, networks.K)
        floor = solve_under_ceilings(
            networks, ceilings, minimize_total=True, func_nonlinear=FUNC_NONLINEAR,
        )
        summaries.append({
            "decay": decay,
            "points": frontier.n_points,
            "elapsed": perf_counter() - solve_start,
            "last_total": sum(last.values()),
            "last_log_sum": log_sum(last),
            "last_minimum": min(last.values()),
            "last_maxima": ceilings,
            "floor_total": sum(floor["capacities"].values()),
        })
    return summaries


def main():
    start = perf_counter()
    networks = load_networks()
    benchmark = load_benchmark()
    fixed = load_frontier()
    moving = load_frontier(filename=MOVING)
    repeated_arc = load_frontier(data_dir=PRE_FIX_DATA_DIR, filename=REPEATED_ARC)
    prefix_budget = sum(benchmark.capacities(18).values())
    fixed_ceilings = layer_maxima(fixed.capacities(0), networks.K)
    moving_ceilings = layer_maxima(moving.capacities(-1), networks.K)

    print("Solving the ceiling floors", flush=True)
    floor = solve_under_ceilings(
        networks, moving_ceilings, minimize_total=True, func_nonlinear=FUNC_NONLINEAR,
    )
    results = {
        "throughput_bounds": layer_throughput_bounds(networks),
        "bottlenecks": layer_bottlenecks(networks),
        "fixed_ceilings": fixed_ceilings,
        "moving_ceilings": moving_ceilings,
        "floor_status": floor["status"],
        "floor_total": sum(floor["capacities"].values()),
        "floor_bound": floor["bound"],
        "floor_log_sum": log_sum(floor["capacities"]),
        "floor_by_layer": layer_totals(floor["capacities"], networks.K),
        "minimum_by_layer": layer_totals(fixed.capacities(0), networks.K),
        "prefix_selection_budget": prefix_budget,
        "reference_minimum_flows": reference_minimum_flows(networks, fixed.capacities(0)),
    }

    print("Checking that fixing the binaries at one keeps the optimum", flush=True)
    free_solve = solve_under_ceilings(
        networks, fixed_ceilings, budget=prefix_budget, func_nonlinear=FUNC_NONLINEAR,
    )
    fixed_binaries = solve_under_ceilings(
        networks, fixed_ceilings, budget=prefix_budget, func_nonlinear=FUNC_NONLINEAR,
        all_functional=True,
    )
    results["binaries_at_one"] = {
        "free_objective": free_solve["objective"],
        "fixed_objective": fixed_binaries["objective"],
        "largest_capacity_difference": max(
            abs(free_solve["capacities"][key] - fixed_binaries["capacities"][key])
            for key in free_solve["capacities"]
        ),
    }

    print("Tightening one layer's ceiling at a time", flush=True)
    results["single_ceiling"] = single_ceiling_effects(
        networks, fixed_ceilings, results["throughput_bounds"], prefix_budget,
    )

    budgets = sorted(GRID + (prefix_budget,))
    rows = {"fixed": [], "moving": []}
    summaries = []
    prefix_capacities = {}
    for rule, ceilings in (("fixed", fixed_ceilings), ("moving", moving_ceilings)):
        for budget in budgets:
            print(f"Matched budget: {rule} at {budget:.4f}", flush=True)
            solve_start = perf_counter()
            solve = solve_under_ceilings(
                networks, ceilings, budget=budget, func_nonlinear=FUNC_NONLINEAR,
            )
            capacities = solve["capacities"]
            summaries.append({
                "rule": rule, "budget": budget, "status": solve["status"],
                "objective": solve["objective"], "bound": solve["bound"],
                "elapsed": perf_counter() - solve_start,
                "total": sum(capacities.values()),
                "log_sum": log_sum(capacities),
                "minimum": min(capacities.values()),
                "maxima": layer_maxima(capacities, networks.K),
                "recheck_failure": recheck_failure(networks, capacities),
            })
            row = {str(key): value for key, value in capacities.items()}
            row["sum_log_u"] = summaries[-1]["log_sum"]
            row["sum_u"] = summaries[-1]["total"]
            rows[rule].append(row)
            summaries[-1]["layer_totals"] = layer_totals(capacities, networks.K)
            if budget == prefix_budget:
                prefix_capacities = dict(prefix_capacities, **{rule: capacities})
    results["matched_budget"] = summaries
    results["gas_route"] = gas_route_flows(
        networks, prefix_capacities["fixed"], prefix_capacities["moving"],
    )

    print("Rechecking the frontiers", flush=True)
    results["recheck"] = {
        "benchmark": recheck_frontier(networks, benchmark),
        "repeated_arc": recheck_frontier(networks, repeated_arc),
        "fixed": recheck_frontier(networks, fixed),
        "moving": recheck_frontier(networks, moving),
    }
    results["decay_sensitivity"] = decay_sensitivity(networks)
    results["elapsed"] = perf_counter() - start

    for rule in rows:
        write_new_csv(pd.DataFrame(rows[rule]), f"matched_budget_{rule}_{TAG}.csv")
    path = os.path.join(DATA_DIR, f"paper_analysis_{TAG}.json")
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print("Wrote", path, f"in {results['elapsed']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
