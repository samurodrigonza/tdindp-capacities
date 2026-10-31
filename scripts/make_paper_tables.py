"""Print the paper's table values from the frontier files; no solver needed.

Run from the repository root, after ``run_paper_analysis.py``::

    python scripts/make_paper_tables.py

The appendix rows are printed ready for LaTeX. All sums are recomputed from the
capacity vectors, never read from a file's stored objective columns.
"""

import json
import os

from tdindp_capacities import (
    DATA_DIR, PRE_FIX_DATA_DIR, load_benchmark, load_frontier,
)

TAG = "2026-10-31"
LAYERS = ("Gas", "Power", "Water")


def number(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", "{,}")


def describe(frontier, row):
    summary = frontier.point_summary(row)
    capacities = frontier.capacities(row)
    summary["maxima"] = {
        k: max(v for key, v in capacities.items() if key[2] == k) for k in LAYERS
    }
    return summary


def show(label, summary):
    maxima = " / ".join(f"{summary['maxima'][k]:.2f}" for k in LAYERS)
    print(f"{label:<28} C={summary['sum_u']:>12,.4f}  F={summary['sum_log_u']:>10,.4f}  "
          f"min={summary['min_capacity']:>8.4f}  max={maxima}")


def main():
    fixed = load_frontier()
    moving = load_frontier(filename=f"pareto_moving_ceiling_{TAG}.csv")
    benchmark = load_benchmark()
    repeated = load_frontier(
        data_dir=PRE_FIX_DATA_DIR, filename=f"pareto_regenerated_repeated_arc_{TAG}.csv",
    )
    with open(os.path.join(DATA_DIR, f"paper_analysis_{TAG}.json"), encoding="utf-8") as stream:
        analysis = json.load(stream)

    print("Named solutions")
    show("Corrected u^C (row 0)", describe(fixed, 0))
    show("Corrected u^B (row 18)", describe(fixed, 18))
    show("Corrected u^NF (row 1)", describe(fixed, 1))
    show("Moving last (row 17)", describe(moving, 17))
    show("Benchmark u^C (row 0)", describe(benchmark, 0))
    show("Benchmark u^P (row 18)", describe(benchmark, 18))
    show("Repeated-arc row 0", describe(repeated, 0))
    show("Repeated-arc row 18", describe(repeated, 18))

    print("\nSweep rows, fixed against moving")
    for row in range(fixed.n_points):
        a = fixed.point_summary(row)
        if row < moving.n_points:
            b = moving.point_summary(row)
            print(f"{row:>3} C={a['sum_u']:>11,.2f}/{b['sum_u']:>11,.2f} "
                  f"F={a['sum_log_u']:>9,.3f}/{b['sum_log_u']:>9,.3f} "
                  f"dF={b['sum_log_u'] - a['sum_log_u']:>8.3f} "
                  f"min={a['min_capacity']:>7.3f}/{b['min_capacity']:>7.3f}")
        else:
            print(f"{row:>3} C={a['sum_u']:>11,.2f}/{'infeasible':>11} "
                  f"F={a['sum_log_u']:>9,.3f}  min={a['min_capacity']:>7.3f}")

    print("\nAppendix rows (increasing total capacity)")
    for row in sorted(range(fixed.n_points), key=lambda r: fixed.point_summary(r)["sum_u"]):
        a = fixed.point_summary(row)
        label = {0: "$u^C$", 1: "$u^{NF}$", 18: "$u^B$"}.get(row, str(row))
        left = f"{label} & {number(a['sum_u'])} & {number(a['sum_log_u'])} & {number(a['min_capacity'])}"
        if row < moving.n_points:
            b = moving.point_summary(row)
            print(f"{left} & {number(b['sum_log_u'])} & {number(b['min_capacity'])} \\\\")
        else:
            print(f"{left} & --- & --- \\\\")

    print("\nSupplementary solves")
    for solve in analysis["matched_budget"]:
        failure = solve["recheck_failure"]
        print(f"{solve['rule']:<7} budget={solve['budget']:>11,.2f} F={solve['log_sum']:>9,.3f} "
              f"bound={solve['bound']:>9,.3f} min={solve['minimum']:>7.3f} "
              f"t={solve['elapsed']:>5.1f}s "
              f"recheck={'pass' if failure is None else failure}")

    print("\nRecheck")
    for name, result in analysis["recheck"].items():
        print(f"{name:<14} {result['passed']}/{result['points']} {result['failures']}")

    print("\nSchedule sensitivity")
    for entry in analysis["decay_sensitivity"]:
        maxima = " / ".join(f"{entry['last_maxima'][k]:.4f}" for k in LAYERS)
        print(f"decay={entry['decay']:<5} points={entry['points']:>3} "
              f"last C={entry['last_total']:>10,.2f} F={entry['last_log_sum']:>9,.2f} "
              f"min={entry['last_minimum']:>6.2f} ceilings={maxima} "
              f"floor={entry['floor_total']:,.2f} t={entry['elapsed']:.0f}s")

    print("\nBounds and floors")
    for k in LAYERS:
        bound = analysis["throughput_bounds"][k]
        print(f"{k:<6} T={bound['bound']:.6f} ({bound['node']}: {bound['availability']} over "
              f"{bound['degree']}) L={analysis['bottlenecks'][k]:.6f} "
              f"fixed={analysis['fixed_ceilings'][k]:.4f} moving={analysis['moving_ceilings'][k]:.4f}")
    print(f"Floor total={analysis['floor_total']:,.4f} bound={analysis['floor_bound']:,.4f} "
          f"F={analysis['floor_log_sum']:,.4f}")


if __name__ == "__main__":
    main()
