# tdindp-capacities

A small package that reads the tdINDP interdependent-network instance
(Gas / Power / Water), generates its capacity frontier with Gurobi, and explains
**where the arc capacities actually come from**.

**Reading existing data is solver-free; generating capacities requires
Gurobi and a valid license.**

> The arc capacities used by the optimisation models are **not** the `u`
> column of `GasArcs.csv`, `PowerArcs.csv` or `WaterArcs.csv`. Those values are
> replaced wholesale by a single row of a pre-computed Pareto frontier.
> **Zero of the 360 arcs keep their CSV value.**

```python
from tdindp_capacities import load_networks, load_capacities

nets = load_networks()
u = load_capacities()     # -> {('Gas_node_9','Gas_node_0','Gas',0): ..., ...}
```

## Which data to use

| Use | File | Read with |
|---|---|---|
| **Everything new** | `data/pareto_regenerated_2026-10-31.csv`, the corrected frontier | `load_frontier()`, `load_capacities()` |
| The moving-ceiling experiment | `data/pareto_moving_ceiling_2026-10-31.csv` | `load_frontier(filename=...)` |
| **Answering questions about published papers** | `pre_fix/data/benchmark_used_for_research_papers.csv` | `load_benchmark()` |

Earlier publications used a frontier generated before the corrections described
below. That data, the code that generated it, and instructions for reproducing it
exactly are preserved in [`pre_fix/`](pre_fix/README.md). If someone asks for the
data behind a published paper, that folder is the answer. Do not use it for new work.

## Contents

| Path | What it is |
|---|---|
| `tdindp_capacities/networks.py` | Solver-free CSV loader: sets, `b`, `c`, `f`, `q`, `gamma` |
| `tdindp_capacities/capacities.py` | Frontier reader, row selection, diagnostics |
| `tdindp_capacities/generation.py` | Gurobi capacity generator, with fixed layer ceilings |
| `tdindp_capacities/moving_ceiling.py` | Experimental sweep whose ceilings follow the previous solve |
| `tdindp_capacities/analysis.py` | Bottlenecks, throughput bounds, single solves, and feasibility rechecks |
| `scripts/` | Regenerates the paper's analysis, table values, and figures |
| `data/` | Instance CSVs, corrected frontier, and experiment outputs |
| `pre_fix/` | Code and data behind earlier publications |
| `example.py` | Runnable, annotated walkthrough of the reader |

```bash
pip install -r requirements.txt
python example.py
```

The reader and `example.py` require only `pandas`.

## The instance

| Network | Nodes | Arcs | Supply nodes | Demand nodes | Total supply | Total demand |
|---|---|---|---|---|---|---|
| Gas   | 16 | 34  | 3  | 13 | 1000 | 1000 |
| Power | 75 | 184 | 9  | 42 | 1000 | 1000 |
| Water | 49 | 142 | 15 | 34 | 1000 | 1000 |

**360 arcs in total**, which is the number of capacity columns in a frontier file.

Arc counts are *post-bidirectionalisation*. Each CSV row defines one directed arc;
the loader then appends the reverse arc wherever it is missing and mirrors `u`, `f`
and `c` onto it. The 181 CSV rows hold 180 unique arcs, because `PowerArcs.csv` lists
arc 5 → 64 on two rows (IDs 40 and 41, with slightly different lengths and costs). The
loader keeps it once, with the later row's `f` and `c`.

A fourth network, Telecommunication, is excluded, as in the original study, and
`Interdep.csv` rows referencing it are skipped.

## Where capacities come from

A frontier CSV has 19 rows and 362 columns: 360 capacity columns headed by the string
repr of a tuple, e.g. `"('Gas_node_9', 'Gas_node_0', 'Gas', 0)"`, plus `sum_u` and
`sum_log_u`. The generator produces it with an **ε-constraint sweep**:

1. Minimize `sum(u)` under the nominal flow constraints. This is row 0.
2. Take each layer's largest capacity from that solution as its ceiling, and maximize
   `sum(log(u))` with `log(u) <= log(ceiling)`. This anchor sets the first budget and is
   not exported.
3. Re-solve with `sum(u) <= previous_total * epsilon`, starting at `epsilon = 1` and
   multiplying by `0.99` each iteration, while the previous total exceeds the minimum
   and epsilon exceeds `0.1`. An infeasible iteration does not end the loop.

| Row | `sum_u` | `sum_log_u` | Thinnest arc | Role |
|---|---:|---:|---:|---|
| 0 | 10,048.79 | 789.62 | **1.00** | Minimum-capacity reference |
| 1 | 50,347.52 | 1,735.04 | 90.30 | Fairness-anchor solution; every arc at its ceiling |
| … | … | … | … | ε sweep |
| 17 | 12,834.07 | 1,197.98 | 20.42 | ε sweep |
| **18** | **10,818.40** | **1,056.96** | **8.82** | **Terminal solution, the default selection** |

The logarithmic objective is motivated by the proportional-fairness and Nash-welfare
ideas discussed by Sampat and Zavala (2019), [*Fairness measures for decision-making and
conflict resolution*](https://doi.org/10.1007/s11081-019-09452-3). Its marginal gain is
`1/u`, so increasing a small capacity is rewarded more than increasing a large one.
These are modeled directed-capacity sums, not monetary investment costs or guarantees
of disruption resilience.

Capacity calibration uses an undisrupted nominal network. Node and arc functionality
binaries remain in the model, and an all-functional assignment satisfies the dependency
constraints for this instance. The layers share an aggregate capacity budget, so their
allocations are not independent design problems.

### Selecting a design

```python
from tdindp_capacities import load_capacities, load_frontier

u = load_capacities()                    # row 18 of the corrected frontier
u = load_capacities(row=-1)              # identical: 18 is the last row
u = load_capacities(row=0)               # the minimum-capacity design
u = load_capacities(collapse_time=True)  # key by (i, j, k), dropping t
u = load_capacities(horizon=[0, 1, 2])   # replicate t=0 across periods

frontier = load_frontier()
frontier.summary_table()        # all 19 points, recomputed from the capacities
frontier.point_summary(18)      # scalar diagnostics for one point
frontier.compare(0, 18)         # per-arc delta between two designs
frontier.as_frame(18)           # tidy tail/head/network/capacity table
```

Row 18 is a position, not a design: another frontier's row 18 is a different vector.
Record which file and row you use.

## Things to know when consuming the data

- **The CSV capacities are decorative.** `compare_to_nominal(load_networks())` shows
  that no arc keeps its CSV value. The `u` column is loaded into `u_nominal` only so
  you can diff against it.
- **`gamma` is keyed by a 5-tuple:** `(dependee, depender, dependee_network,
  depender_network, t)`. `i` is the dependee (the provider) and `j` the depender.
  `load_networks()` returns a `Gamma` dict whose `in`, `[]` and `.get()` raise
  `ValueError` for any other key shape. A plain dict assigned to a hand-built `Networks`
  is not guarded.
- **Time is collapsed.** The study is single-period, `T = [0]`, and the frontier stores
  only `t = 0`. `load_capacities(horizon=[...])` replicates it, since capacities are
  static. The generator rejects other horizons.

## Generating a frontier

Use a compatible Python/Gurobi installation and a valid license:

```powershell
python -m pip install -e ".[gurobi]"
```

```python
from tdindp_capacities import build_capacity_frontier, load_networks

frontier = build_capacity_frontier(
    load_networks(), filename="my_frontier.csv", func_nonlinear=1,
)
print(frontier.summary_table())
```

The output has the same CSV format as the shipped frontier. An existing output file is
rejected, never overwritten. `epsilon`, `epsilon_decay`, and `epsilon_floor` expose the
sweep constants. Arcs must be unique, as `load_networks()` supplies them.

`func_nonlinear` makes the logarithm representation explicit for every solve:

| Value | Behavior |
|---|---|
| `None` (default) | Do not override the solver's log handling |
| `0` | Static piecewise-linear approximation (Gurobi 11+) |
| `1` | Nonlinear function constraints, used for all shipped results (Gurobi 11+) |

Use `func_nonlinear=1`. In default mode on Gurobi 11.0.3, the minimum-capacity solve
expanded to 21,505 presolved columns and remained at a 0.12% gap after an hour. The
relative MIP gap is never overridden: its effective default is `0.0001` (0.01%).
"Nonlinear" describes the function representation, not infinite-precision arithmetic or
a zero gap, and solver tolerances still apply.

### Logs

Each invocation creates a unique, UTC-prefixed run directory beside the output CSV, or
under `log_dir`. High-level progress also goes to the console.

| File | Contents |
|---|---|
| `progress.log` | Stages, environment versions, settings, statuses, objectives, and errors |
| `minimum-anchor.log`, `fairness-anchor.log`, `sweep-0001.log`, ... | Native Gurobi progress per solve |
| Matching `.prm` files | Nondefault parameters recorded before each solve |

Logs are diagnostic records, not restart checkpoints: accepted capacity vectors stay in
memory until the final CSV export, and only `Run completed` confirms a successful run.
Default log directories are ignored by Git. Review logs before committing them, since
they can contain local paths and environment details.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Solver checks are skipped when `gurobipy` is not installed.

---

## Patch: 31 October 2026

### Files

| Before | After | Note |
|---|---|---|
| `data/pareto_of_capacities.csv` | `pre_fix/data/benchmark_used_for_research_papers.csv` | Byte-for-byte unchanged; no longer the default |
| n/a | `data/pareto_regenerated_2026-10-31.csv` | Corrected frontier; **now the default** |
| n/a | `data/pareto_moving_ceiling_2026-10-31.csv`, `_trace.csv` | Moving-ceiling experiment |
| n/a | `pre_fix/` | Old code and data, kept for published papers |

`load_capacities()` and `load_frontier()` now return the **corrected** values. Row 18
totals 10,818.40, not 11,440.77. Code that needs the published values must call
`load_benchmark()`. Scripts that opened `data/pareto_of_capacities.csv` directly must be
updated.

### Corrections

- **Dependency constraints** (in the code since commit `1b0ce27`): the original guard
  used an unrelated node index and built no inequality. The generator now uses the
  actual providers: 21 grouped inequalities from 27 relationships.
- **Repeated arc:** `load_networks()` lists arc 5 → 64 once, and the generator rejects
  repeated arcs. Each arc has one capacity variable and weight one in every objective.
- **Stored objectives:** `sum_u` and `sum_log_u` are computed from the capacities, and
  agree with them to about `1e-11`. `point_summary()` and `summary_table()` always
  recompute. `point_summary()` no longer has a `recomputed_sum_log_u` key.
- **`gamma` keys:** `Gamma` raises on any lookup whose key is not a 5-tuple.
- **Labels:** `summary_table()` no longer calls row 1 "unconstrained", or the last row
  "the one used".

The repeated arc was the single cause of both long-standing objective mismatches in the
benchmark: −11.97 in row 0's `sum_u`, and +13.8155 in the later rows' `sum_log_u`. After
the fix, a full regeneration has neither.

### Corrected frontier

Run with Python 3.9.12, Gurobi 11.0.3, `FuncNonlinear=1`, and default tolerances on an
AMD Ryzen 3 2300U with 4 logical processors and 12.0 GB RAM. Three runs produced identical
capacity vectors, in 216.94 to 268.51 seconds.

| Quantity | Published benchmark | Dependency fix only | Corrected |
|---|---:|---:|---:|
| Solutions | 19 | 19 | 19 |
| Minimum total capacity | 10,048.79 | 10,048.79 | 10,048.79 |
| Reference Gas / Power / Water maximum | 420.70 / 125.20 / 95.48 | 420.70 / 124.44 / 115.60 | 420.70 / 126.20 / 90.30 |
| Row 18 total capacity | 11,440.77 | 12,001.47 | 10,818.40 |
| Row 18 sum of log capacities | 1,114.23 | 1,152.56 | 1,056.96 |
| Row 18 smallest arc | 12.85 | 16.31 | 8.82 |
| Fixed-capacity recheck | 19/19 | 18/19 | 18/19 |

Row 18 is not comparable across columns, because each column's budget schedule lands
differently. Solving the corrected model at the benchmark's budget of 11,440.77 gives
a sum of log capacities of 1,114.34 and a smallest arc of 13.06, with 170 of 180 connections
within 5% of the published vector. The corrected recheck failure is row 17, with a
reverse-direction mismatch of about `4.4e-6` that clears with a capacity allowance of `1e-5`.
No capacities were adjusted.

### Experiment: moving log-capacity ceilings

`build_capacity_frontier()` sets each layer's `log(u)` ceiling once, from the minimum
solve. `build_moving_ceiling_frontier()` keeps that for the fairness anchor and first
budgeted solve, then caps each later solve at the layer maxima of the last feasible
solve. A solution cannot exceed its ceiling, so the ceilings can only fall. This is an
experiment; the default generator is unchanged. Its run took 202.83 seconds over 230
budgeted solves and exported 18 solutions.

| Common budget | `sum_log_u`: fixed → moving | Smallest arc: fixed → moving | Gas maximum: fixed → moving |
|---:|---:|---:|---:|
| 35,062.68 | 1,639.80 → 1,639.80 | 90.30 → 90.30 | 351.35 → 351.35 |
| 22,989.17 | 1,474.69 → 1,474.29 | 56.60 → 56.16 | 379.30 → 351.35 |
| 15,073.06 | 1,287.58 → 1,282.50 | 29.85 → 28.26 | 401.34 → 351.35 |
| 12,834.07 | 1,197.98 → 1,183.09 | 20.42 → 17.07 | 410.52 → 351.35 |
| 11,440.77 | 1,114.34 → 1,069.13 | 13.06 → 7.91 | 417.11 → 351.35 |
| 10,900.00 | 1,065.91 → 945.46 | 9.42 → 1.63 | 418.47 → 351.35 |
| 10,818.40 | 1,056.96 → *infeasible* | 8.82 → *infeasible* | 418.77 → *infeasible* |

- **The ceilings stop at a structural bound.** They reach Gas 351.35, Power 86.00, and
  Water 55.167, which equal each layer's largest `|b| / degree` over its nodes (Gas node 1:
  702.7 / 2; Power node 1: 258 / 3; Water node 16: 55.167 / 1). A linear program confirms
  that no lower layer maximum can serve demand. `analysis.py` computes both.
- **The minimum achievable total rises** to 10,857.36, which is 8.05% above 10,048.79. Gas
  accounts for 475.20 of that rise, Power for 323.87, and Water for 9.51.
- **The Gas ceiling costs the most.** Tightened alone at 11,440.77, it lowers `sum_log_u` by 24.86
  of the joint 45.21, and takes about 200 from each of Power and Water through the shared budget.
- **The sweep stops early** only because its next budget (10,818.40) falls below that floor,
  after which all 213 remaining solves are infeasible. The 11,440.77 and 10,900 rows above
  come from direct solves, because the schedule steps over those budgets.
- **The step size does not change this.** With budget decay 0.98, 0.95, and 0.90, the ceilings
  and the floor are the same, although the last retained solution differs.
- **The trade-off:** smaller largest arcs, in exchange for a lower sum of log capacities, a
  thinner smallest arc, and a higher minimum total. These results concern nominal
  capacities only, not disruption performance.

The moving-ceiling recheck passes 14/18; rows 14–17 have reverse-direction mismatches of
`1.1e-6` to `2.4e-6`, which clear with allowances of `1e-7` to `1e-6`.

### Note on the model class

Without damage, any feasible point stays feasible with every functionality binary at one, so the
solves have the same optimal values as a continuous convex model. `sum(log(u))` is strictly concave
in `u`, so each budgeted solve's capacity vector is unique for given ceilings and budget; only the
minimum solve, then a linear program, can have alternative optima. A check at 11,440.77 gave the
same objective to within `1e-5` and the same capacities to within `1.1e-6`. This does not extend to
damaged networks.

### Reproducing the paper's numbers

```powershell
python -m pip install -e ".[gurobi,figures]"
python scripts/run_paper_analysis.py            # needs Gurobi; about 7 minutes
python scripts/make_paper_tables.py             # prints the table values
python scripts/make_paper_figures.py figures    # writes the figures and arc statistics
```

`run_paper_analysis.py` refuses to overwrite its outputs, so remove them first to rerun it.
