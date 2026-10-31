# Pre-fix code and data

**Do not use anything in this folder for new work.** It exists because earlier
publications used capacity data generated before the corrections described in
the [main README](../README.md). If someone asks for the data or code behind one
of those publications, this is what to give them. Nothing here should be edited.

## What is here

| Path | What it is |
|---|---|
| `data/benchmark_used_for_research_papers.csv` | **The frontier used in the published papers**, byte-for-byte as it was. Named `pareto_of_capacities.csv` in the original code. Published work selects its row 18. |
| `data/benchmark_reproduction_gurobi_9.1.2.csv` | A rerun of the published frontier from `code_published_frontier/`, made on 31 October 2026 (see below). |
| `data/pareto_regenerated_repeated_arc_2026-10-31.csv` | The output of `code_dependency_fix/`: dependency guard corrected, repeated arc still present. |
| `code_published_frontier/` | Snapshot of commit `e182e7c`, the extraction that reproduces the published frontier. |
| `code_dependency_fix/` | Snapshot of commit `7841097`, which corrected the dependency guard only. |

Each snapshot is a complete, self-contained copy of the repository at that commit,
including its own README, tests, and instance data under its original file names.
Its `data/pareto_of_capacities.csv` is identical to the benchmark above. Solver
logs for the two reruns sit beside the CSVs in `*_logs/` folders, which Git ignores.

From the main package, read the benchmark with:

```python
from tdindp_capacities import load_benchmark

benchmark = load_benchmark()
u = benchmark.capacities(row=18)   # the published selection
```

## Reproducing the published frontier

The published frontier can be regenerated exactly. On 31 October 2026, the code in
`code_published_frontier/` reproduced **all 19 solutions and all 360 arcs**. The
largest absolute difference across the whole file is `4.96e-8`, which is the
rounding of the stored values. The run took 436.34 seconds.

| Component | Configuration |
|---|---|
| Code | `code_published_frontier/` (commit `e182e7c`) |
| Python | 3.8.0, with pandas 1.2.4 |
| Solver | Gurobi 9.1.2, build `v9.1.2rc0`, 4 threads |
| Solver settings | Defaults: no `FuncNonlinear` (it does not exist before Gurobi 11), default gap and tolerances |
| Processor | AMD Ryzen 3 2300U with Radeon Vega Mobile Gfx, 2.00 GHz, 4 logical processors |
| Memory | 12.0 GB installed (10.9 GB usable) |
| Storage | 932 GB SSD (CT1000BX500SSD1) |
| Graphics | AMD Radeon Vega 6 Graphics (1010 MB) |
| System | 64-bit operating system, x64-based processor; Windows 10 |

To repeat it from this folder, with that Python and solver:

```powershell
cd pre_fix\code_published_frontier
python -c "from tdindp_capacities import build_capacity_frontier, load_networks; build_capacity_frontier(load_networks(), data_dir='rerun', filename='rerun.csv')"
```

The original solver logs from the published run were not retained, so this
establishes a configuration that reproduces the file, not a record of the original
run. Gurobi 9.1.2 and Python 3.8.0 are still installed on the machine named in the
provenance record. The same code under Gurobi 11.0.3 in default mode did not finish a
minimum-capacity solve within an hour.

## How this code differs from the current code

| Item | This code | Current code |
|---|---|---|
| Dependency constraints | `code_published_frontier/`: an unrelated node index guarded construction, so **no** dependency inequality was built. `code_dependency_fix/`: corrected. | Actual provider sets: 21 grouped inequalities from 27 relationships. |
| Repeated arc | `PowerArcs.csv` lists arc 5 -> 64 on two rows, and the model created duplicate variables for it, counting it twice in the minimum objective. | One capacity decision per directed pair. |
| Stored objectives | `sum_u` is off by -11.97 in row 0, and `sum_log_u` by a constant +13.8155 in rows 1-18. Both trace to the repeated arc: its capacity, counted twice, and `-log(1e-6)` from a leftover variable at zero. | Computed from the capacities; they agree to about `1e-11`. |
| Logarithm | Default piecewise-linear approximation in Gurobi 9.1.2. The solver logs report violations of the logarithm constraints in 19 solves, from `9.93e-6` to `5.65`. | Nonlinear function constraints in Gurobi 11.0.3 (`FuncNonlinear=1`). |

The capacity columns of the benchmark are internally consistent, and readers should
recompute `sum_u` and `sum_log_u` from them rather than quote the stored columns.
`point_summary()` and `summary_table()` do this automatically.

## Checks on this data

These describe the data; they are not a reassessment of any published result.

- **Feasibility.** With each capacity vector fixed in the corrected model and all
  functionality states set to one, the benchmark passes **19 of 19** solutions at the
  default feasibility tolerance. The repeated-arc rerun passes 18 of 19; its row 18
  has a reverse-direction mismatch of about `3.7e-6`.
- **Closeness to the corrected model.** Re-solving the corrected model at the budget
  of the published selection (total capacity 11,440.77) gives a sum of log capacities
  of 1,114.34, against 1,114.23 for the published vector. Of 180 connections, 170
  are within 5% of the published value, and the largest difference is 10.74%.
- **Repeated-arc rerun across machines.** `code_dependency_fix/` on this machine (Python
  3.9.12, Gurobi 11.0.3, `FuncNonlinear=1`) matches the figures recorded for its earlier
  run on an Intel Core Ultra 7 165H (Python 3.8.0, same solver): 19 solutions, minimum
  total 10,048.78746155, index-18 total 12,001.4659, smallest arc 16.31.
