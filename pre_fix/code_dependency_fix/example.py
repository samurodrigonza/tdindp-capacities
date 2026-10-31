"""Worked example: how to read the tdINDP arc capacities.

Run from the repository root::

    python example.py

Every number printed below is read from ``data/``; nothing is solved and
Gurobi is not required.
"""

import pandas as pd

from tdindp_capacities import (
    DEFAULT_ROW,
    compare_to_nominal,
    load_capacities,
    load_frontier,
    load_networks,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
rule("1. The instance")
# ---------------------------------------------------------------------------
nets = load_networks()
print(nets.summary().to_string(index=False))
print("\nArc counts are post-bidirectionalisation: each CSV row becomes two")
print("directed arcs, which is why there are 360 capacity keys in total.")

# ---------------------------------------------------------------------------
rule("2. The capacity frontier")
# ---------------------------------------------------------------------------
frontier = load_frontier()
print("file      :", frontier.path)
print("points    :", frontier.n_points, "(rows 0..%d)" % (frontier.n_points - 1))
print("arcs/point:", frontier.n_arcs)
print()
print(frontier.summary_table().to_string(index=False))
print()
print("Row 0 is the cost-optimal reference. Rows 1..18 are the epsilon sweep:")
print("row 1 maximises sum_log_u unconstrained, then the total-capacity budget")
print("is tightened by a factor of 0.99 per iteration until the sweep stops.")

# ---------------------------------------------------------------------------
rule("3. Reading the capacities the models actually use")
# ---------------------------------------------------------------------------
u = load_capacities()  # defaults to row 18 -- the shipped behaviour
print("load_capacities() -> %d entries keyed by (tail, head, network, t)" % len(u))
for key in list(u)[:5]:
    print("   ", key, "->", u[key])

print("\nIdentical selections (row 18 is the last row):")
print("   load_capacities(row=%d) == load_capacities(row=-1) :" % DEFAULT_ROW,
      load_capacities(row=DEFAULT_ROW) == load_capacities(row=-1))

print("\nSingle-period models drop the trailing t:")
u_flat = load_capacities(collapse_time=True)
print("   ", list(u_flat)[0], "->", list(u_flat.values())[0])

print("\nPoint diagnostics:")
for name, value in frontier.point_summary().items():
    print("    %-22s %s" % (name, value))

# ---------------------------------------------------------------------------
rule("3b. The two objective columns do not reconcile -- recompute them")
# ---------------------------------------------------------------------------
check = frontier.objective_consistency()
print(check[["row", "sum_u_error", "sum_log_u_error"]].to_string(index=False))
print()
print("Row 0's sum_u is off by %+.4f from its own capacity columns." % check["sum_u_error"].iloc[0])
print("Rows 1-18 carry a CONSTANT sum_log_u offset of %+.4f." % check["sum_log_u_error"].iloc[1])
print("The capacity columns are self-consistent and are all that gets used;")
print("treat sum_u / sum_log_u as provenance notes, not as reportable numbers.")

# ---------------------------------------------------------------------------
rule("4. The CSV capacities are NOT the model capacities")
# ---------------------------------------------------------------------------
cmp = compare_to_nominal(nets)
unchanged = (cmp["delta"].abs() < 1e-9).sum()
print("arcs compared          :", len(cmp))
print("arcs keeping CSV value :", unchanged)
print("largest increase       : %+.2f" % cmp["delta"].max())
print("largest decrease       : %+.2f" % cmp["delta"].min())
print("\nMost-changed arcs:")
print(cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index)
         .head(8).to_string(index=False))
print("\nThe selected capacities come from the calibrated frontier, not the source arc files.")

# ---------------------------------------------------------------------------
rule("5. Fairness is what row 18 buys you")
# ---------------------------------------------------------------------------
diff = frontier.compare(0, DEFAULT_ROW)
print("Row 0 (cost-optimal) vs row %d (used):" % DEFAULT_ROW)
print("   thinnest arc : %.4f  ->  %.4f" % (
    frontier.point_summary(0)["min_capacity"],
    frontier.point_summary(DEFAULT_ROW)["min_capacity"]))
print("   sum_u        : %.2f  ->  %.2f" % (
    frontier.point_summary(0)["sum_u"],
    frontier.point_summary(DEFAULT_ROW)["sum_u"]))
print("   sum_log_u    : %.2f  ->  %.2f" % (
    frontier.point_summary(0)["sum_log_u"],
    frontier.point_summary(DEFAULT_ROW)["sum_log_u"]))
print("\nArcs pushed up the most by the fairness objective:")
print(diff.tail(5).to_string(index=False))

# ---------------------------------------------------------------------------
rule("6. Interdependencies: mind the 5-tuple key")
# ---------------------------------------------------------------------------
print("gamma entries:", len(nets.gamma))
example = next(iter(nets.gamma))
print("example key  :", example)
dependee, depender, k, k_tilde, t = example
print("\nA 4-tuple lookup returns nothing and raises nothing:")
print("   (dependee, depender, k, k_tilde) in gamma ->",
      (dependee, depender, k, k_tilde) in nets.gamma)
print("   full 5-tuple in gamma                     ->", example in nets.gamma)
print("\nThat silent miss once disabled every interdependency constraint in the")
print("downstream models. Always key gamma with the trailing t.")

rule("done")
