# tdindp-capacities

A small package that reads the tdINDP interdependent-network instance
(Gas / Power / Water), optionally generates its capacity frontier with
Gurobi, and explains **where the arc capacities actually come from**.

**Reading existing data is solver-free; generating capacities requires
Gurobi and a valid license.**

It exists because that answer is genuinely surprising:

> The arc capacities used by the optimisation models are **not** the `u`
> column of `GasArcs.csv`, `PowerArcs.csv` or `WaterArcs.csv`. Those values
> are read, then discarded and replaced wholesale by a single row of a
> pre-computed Pareto frontier, `pareto_of_capacities.csv`.
> **Zero of the 360 arcs keep their CSV value.**

The source files provide the network topology and nominal capacity values;
the default capacity selection comes from the calibrated frontier instead.
This distinction documents the values actually used, not a general
assessment of the source dataset's reliability.

```python
from tdindp_capacities import load_networks, load_capacities

nets = load_networks()
u = load_capacities()     # -> {('Gas_node_9','Gas_node_0','Gas',0): 47.0, ...}
```

---

## Contents

| path | what it is |
|---|---|
| `tdindp_capacities/networks.py` | solver-free rewrite of the CSV loader: sets, `b`, `c`, `f`, `q`, `gamma` |
| `tdindp_capacities/capacities.py` | **the capacity story** — frontier reader, row selection, diagnostics |
| `tdindp_capacities/generation.py` | optional Gurobi capacity generator with corrected dependency indexing |
| `data/` | the instance CSVs plus `pareto_of_capacities.csv` |
| `example.py` | runnable, annotated walkthrough of everything below |

```bash
pip install -r requirements.txt
python example.py
```

The reader and `example.py` require only `pandas`: no Gurobi or license.
Running `build_capacity_frontier()` additionally requires `gurobipy` and
a valid Gurobi license; installation and usage are documented below.

---

## The instance

| network | nodes | arcs | supply nodes | demand nodes | total supply | total demand |
|---|---|---|---|---|---|---|
| Gas   | 16 | 34  | 3  | 13 | 1000 | 1000 |
| Power | 75 | 185 | 9  | 42 | 1000 | 1000 |
| Water | 49 | 142 | 15 | 34 | 1000 | 1000 |

**360 arcs in total**, and that number matters — it is exactly the number
of capacity columns in the frontier file.

Arc counts are *post-bidirectionalisation*. Each CSV row defines one
directed arc; the loader then appends the reverse arc wherever it is
missing and mirrors `u`, `f` and `c` onto it. 180 CSV rows become 360
directed arcs. Anything that iterates arcs sees the doubled set.

A fourth network, Telecommunication, ships in `data/` but is excluded —
the original loader has it commented out, and `Interdep.csv` rows
referencing it are skipped by a `if ... in K` guard.

---

## Where capacities come from

### The two-mode function

In the original code this all lives in `find_capacities()`, gated by a
hard-coded flag:

```python
def find_capacities(...):
    build = False          # <-- the flag
    if build:
        ...                # solve, then WRITE pareto_of_capacities.csv
    else:
        df = pd.read_csv('pareto_of_capacities.csv')
        df = df.drop(df.columns[0], axis=1)
        row_number = 18    # <-- the selection
        row = df.iloc[row_number]
        u = {ast.literal_eval(col): val for col, val in row.items()
             if col not in ("sum_log_u", "sum_u")}
        return u
```

* **`build = True`** — solves a bi-objective capacity-design problem and
  regenerates the CSV. Needs Gurobi, and is slow.
* **`build = False`** — the shipped setting. Pure CSV read. This is the
  path this package reproduces, exactly.

Because the flag is a literal inside the function body, it is not part of
any run configuration, is not logged, and does not appear in the
experiment provenance. Neither does `row_number`.

### The frontier file

`pareto_of_capacities.csv` is **19 rows × 362 columns**:

* 360 arc-capacity columns, each headed by the *string repr of a Python
  tuple*, e.g. `"('Gas_node_9', 'Gas_node_0', 'Gas', 0)"`. They are parsed
  back with `ast.literal_eval`. The trailing `0` is the time index.
* `sum_u` — total capacity of that design.
* `sum_log_u` — sum of log capacities, i.e. the proportional-fairness
  measure applied to the capacities themselves.
* plus a leading unnamed pandas index column, which is dropped on read.

### How the frontier was generated

An **ε-constraint sweep** over the two objectives:

1. **Row 0** is the *cost-optimal reference*: the capacity design a
   plain minimum-cost solve produces. Cheap in total (`sum_u` = 10,060.76)
   but brutally uneven — its thinnest arc is **exactly 1.0 unit**, and
   `sum_log_u` is only 790.43.
2. **Row 1** maximises `sum_log_u` with no budget cap at all:
   `sum_u` = 50,889.06, `sum_log_u` = 1727.60, thinnest arc 95.48. Fair,
   and about five times more total capacity than the cost-optimal design.
3. **Rows 2–18** re-solve repeatedly, each time adding
   `sum_u <= previous_total * epsilon` with `epsilon` shrinking by a
   factor of 0.99 per iteration, walking the frontier back down toward the
   cost-optimal budget. Duplicate solutions are not appended.

The loop is written to stop once total capacity reaches the cost-optimal
level, but it never gets there — it runs out of ε first (the guard is
`epsilon > 1e-1`). So **row 18 is simply the last point the sweep managed
to solve.**

| row | `sum_u` | `sum_log_u` | thinnest arc | role |
|---|---|---|---|---|
| 0  | 10,060.76 | 790.43  | **1.00**  | cost-optimal reference |
| 1  | 50,889.06 | 1727.60 | 95.48 | unconstrained max fairness |
| …  | …         | …       | …     | ε sweep |
| 17 | 13,557.17 | 1216.80 | 21.88 | ε sweep |
| **18** | **11,440.77** | **1100.41** | **12.85** | **tightest point — the one used** |

### Why row 18

Compare row 18 against row 0, which is the design you would get if you
ignored fairness entirely:

|  | row 0 (cost-optimal) | row 18 (used) |
|---|---|---|
| total capacity `sum_u` | 10,060.76 | 11,440.77 (**+13.7%**) |
| fairness `sum_log_u` | 790.43 | 1100.41 (**+39.2%**) |
| thinnest arc | 1.00 | 12.85 (**12.8×**) |

Row 18 is the knee: it buys a large fairness gain and, crucially,
eliminates the near-zero-capacity arcs, for a modest increase in total
installed capacity. Those 1.0-unit arcs in row 0 are the real problem —
they act as hard bottlenecks and would dominate any downstream
flow-allocation result for reasons that have nothing to do with the
research question.

So `row_number = 18` is a defensible modelling choice. It is just an
**undocumented** one.

### Selecting it in this package

```python
from tdindp_capacities import load_capacities, load_frontier, DEFAULT_ROW

u = load_capacities()                    # row 18 — matches the original bit for bit
u = load_capacities(row=-1)              # identical: 18 IS the last row
u = load_capacities(row=0)               # the cost-optimal design, for contrast
u = load_capacities(collapse_time=True)  # key by (i, j, k), dropping t
u = load_capacities(horizon=[0, 1, 2])   # replicate t=0 across periods

frontier = load_frontier()
frontier.summary_table()        # all 19 points at a glance
frontier.point_summary(18)      # scalar diagnostics for one point
frontier.compare(0, 18)         # per-arc delta between two designs
frontier.as_frame(18)           # tidy tail/head/network/capacity table
```

Prefer `row=-1` or an explicit, recorded `row=` over inheriting the
literal `18`: if the frontier were ever regenerated with a different
iteration count, `18` would silently select a different design while
`-1` would keep its meaning.

---

## Known data quirks

These are all real, reproducible against the shipped `data/`, and worth
knowing before you report anything.

### 1. The CSV capacities are decorative

```python
from tdindp_capacities import load_networks, compare_to_nominal
cmp = compare_to_nominal(load_networks())
(cmp["delta"].abs() < 1e-9).sum()      # -> 0 out of 360
```

Not a single arc keeps the capacity printed in its network CSV. The `u`
column in `GasArcs.csv` and friends is loaded into `u_nominal` here purely
so you can diff against it.

### 2. `sum_u` and `sum_log_u` do not reconcile with the capacity columns

Run `frontier.objective_consistency()`:

* **Row 0**: `sum_log_u` matches its capacity columns exactly, but `sum_u`
  is off by **−11.97**. Row 0's `sum_u` was taken from a *different* solve
  than the capacity vector stored alongside it.
* **Rows 1–18**: `sum_u` matches exactly, but `sum_log_u` is off by a
  **constant +13.8155** every single row.

A constant offset across 18 rows is a bookkeeping artefact in the
generating loop, not numerical noise. The capacity columns themselves are
internally consistent, and they are the only thing `load_capacities()`
reads — so **no downstream result is affected**. But do not quote `sum_u`
or `sum_log_u` from this file. Recompute them from the capacities.

### 3. `gamma` is keyed by a 5-tuple

```python
(dependee, depender, dependee_network, depender_network, t)
```

Look it up with a 4-tuple and Python returns "not present". No exception,
no warning. This exact mismatch once silently disabled **every**
interdependency constraint in the downstream models — the constraints were
being built inside `if gamma.get(4_tuple)` guards that were never true.
Nothing crashed; the models just solved a different problem than intended.

If you write anything that consumes `gamma`, assert on the key length
first. `example.py` §6 demonstrates the failure mode.

Also note the semantics: `i` is the **dependee** (the provider), `j` is the
**depender**. At least one comment in the original source states this
backwards.

### 4. Time is collapsed

The study is single-period, `T = [0]`. The frontier only ever stores
`t = 0`. For multi-period use, `load_capacities(horizon=[...])` replicates
`t = 0` across periods, which is what the original code does — capacities
are static.

---

## Reproducing the original call exactly

```python
# original
import tdINDP_parameters_build as tb
u = tb.find_capacities()            # build = False branch

# here
from tdindp_capacities import load_capacities
u = load_capacities()               # same 360 keys, same values
```

Both return `{(tail, head, network, t): capacity}` with 360 entries.

## Regenerating the frontier

Use `build_capacity_frontier()` with Gurobi and a valid license, as described
below. It implements the original `build = True` path for the shipped nominal
configuration, with the dependency-indexing correction documented below.
Record the resulting row count and your selected row somewhere
durable, and select explicitly from the generated frontier rather than
inheriting the historical default row.

---

## Added: cleaned source capacity generator

The historical walkthrough above is retained. Regeneration is now available
through `tdindp_capacities/generation.py`. Reading the shipped frontier still
requires only pandas and does not import Gurobi.

This module is based on the `build = True` branch of the original
`find_capacities()` routine in `tdINDP_parameters_build.py`,
using its shipped nominal configuration: `T = [0]`, no damaged components,
and an empty critical-demand-node set. It is not the downstream
disruption/restoration optimizer; other horizons are rejected explicitly.

Capacity calibration uses an undisrupted nominal network with all components
available; it does not simulate component failures. Node and arc functionality
binaries remain in the model rather than being fixed to one. Functional
dependency constraints are included, and an all-functional assignment
satisfies them for this instance. The layers share an aggregate capacity
budget, so their capacity allocations are not independent design problems.

### Why perform the multiobjective sweep?

Minimizing the sum of capacities finds a low-total-capacity network that
can carry the nominal flows. However, alternative routes can be left with
near-zero capacity because they are unnecessary for that particular nominal
flow. Those bottlenecks can unduly restrict later flow-allocation experiments.

The logarithmic objective is motivated by the proportional-fairness and
Nash-welfare ideas discussed by Apoorva M. Sampat and Victor M. Zavala
(2019), [*Fairness measures for decision-making and conflict resolution*,
Optimization and Engineering, 20, 1249–1272](https://doi.org/10.1007/s11081-019-09452-3).
For positive allocations, maximizing their product is equivalent to
maximizing the sum of their logarithms. Here, that idea is adapted to arc
capacities as a balancing criterion, not a claim of fairness among people
or a guarantee of equal capacities.

The second objective therefore maximizes `sum(log(u))`. Its marginal gain is `1/u`,
so increasing a small capacity is rewarded more than increasing an already
large one. The sweep exposes how much additional aggregate capacity this
balancing requires instead of choosing an arbitrary weighted sum:

1. Minimize `sum(u)` under the nominal flow constraints.
2. For each layer, take its maximum capacity from that solution as an upper
   bound. Maximize `sum(log(u))` with `log(u) <= log(layer_max)`.
3. Re-solve the log objective with `sum(u) <= previous_total * epsilon`,
   starting at `epsilon = 1` and multiplying epsilon by `0.99` each iteration.
   Retain distinct feasible solutions while the previous feasible total
   exceeds the minimum objective and epsilon exceeds `0.1`. An infeasible
   iteration does not end the loop: epsilon still decreases and the previous
   feasible total remains unchanged, exactly as in the source.

The original excess-supply and unmet-demand variables are retained.
Unmet demand is explicitly fixed to zero. Excess supply is not independently
fixed or minimized in this routine; with balanced nominal supply and demand,
zero excess follows from flow conservation and nonnegativity, up to solver
tolerances. Reverse directions share capacity. These are modeled
directed-capacity sums, not monetary investment costs or guarantees of
disruption resilience.

### Run the generator

Use a compatible Python/Gurobi installation and a valid Gurobi license.
No Gurobi version or log-approximation setting is imposed by this extraction.
To retain an existing solver version, install that version first. From a
checkout of this repository:

```powershell
python -m pip install -e ".[gurobi]"
```

```python
from tdindp_capacities import build_capacity_frontier, load_networks

frontier = build_capacity_frontier(
    load_networks(),
    filename="pareto_regenerated.csv",
)
print(frontier.summary_table())
u = frontier.capacities(row=-1)
```

The output has the existing reader's CSV format: capacity tuple columns,
`sum_u`, `sum_log_u`, and the pandas index. An existing output file is
rejected, never overwritten. `load_capacities()` continues to read the
unchanged historical frontier and select row 18; a new frontier must be
selected explicitly through the returned object or `load_frontier(filename=...)`.
The optional arguments `epsilon`, `epsilon_decay`, and `epsilon_floor`
expose the original sweep constants.

### Source cleanup and reproducibility

The extraction provides module organization, formatting, lazy solver imports,
resource disposal, explicit failure reporting, and a safe output filename.
The dependency guard is corrected as described below; otherwise, the source
formulation and historical bookkeeping are retained:

- Node/arc binary variables, the three flow/functionality capacity
  constraints, excess/shortage variables, and the unoptimized total-cost
  equality remain. Only restoration blocks with no entries in the shipped
  single-period nominal configuration are omitted.
- Capacity and logarithm variables keep their original default nonnegative
  bounds. Log constraints remain in the minimum-capacity model as well as
  the fairness models; no explicit `u >= 1` constraint is added.
- Repeated endpoint entries are not deduplicated. Variable creation,
  dictionary overwriting, `getVarByName` lookups, and extraction from
  `getVars()` follow the source. This intentionally retains its distinction
  between repeated model variables and unique exported capacity keys.
- The first solution's `sum_u` is the minimum model's objective, and its
  `sum_log_u` is computed from the extracted capacity dictionary. Later
  solutions sum all capacity-named model variables for `sum_u` and use
  `log(value + 1e-6)` for `sum_log_u`, exactly as the source does.
- The source sets `OutputFlag = 0` and `InfUnbdInfo = 1`. The generator
  now enables file logging (`OutputFlag = 1`, `LogToConsole = 0`, and a
  per-solve `LogFile`) while retaining `InfUnbdInfo = 1`. By default,
  logarithm handling, feasibility tolerances, and other mathematical solver
  settings remain unchanged. Explicit log-mode selection is available as
  documented below.
- The original dependency guard used a node variable left over from the
  preceding arc loop and constructed no dependency constraints for this
  instance. The generator now collects the actual providers for each
  dependent node, source layer, and time before imposing the weighted
  functionality inequality. The 27 dependency relationships produce 21
  grouped constraints. This is an explicit correction, not a claim that
  the historical generation enforced those constraints.
- The loop keeps the original exact total comparison, multiplicative epsilon
  schedule, and dictionary-equality duplicate check. There is no new
  budget-based early stop. Infeasible iterations emit a warning and continue;
  failed anchors and unexpected solver failures raise an error.

This is source fidelity, not a guarantee of bit-for-bit reproduction of an
old run on another solver installation. Multiple minimum-capacity solutions
can yield different layer maxima, which then set the later logarithm bounds.
No archived-versus-regenerated output comparison is performed by the tests.

A few clarifications to the retained historical narrative above: the
unbudgeted fairness anchor initializes the sweep but is not separately
exported; the first exported fairness point is a budgeted re-solve. The
archived minimum vector sums to approximately `10,048.79`, rather than its
stored `sum_u = 10,060.76`; the discrepancy matches the repeated endpoint's
capacity, not evidence of a separate solve. The source does not establish
that row 18 is a universal knee or record a formal selection policy.

Focused checks cover model construction, sweep control flow, failure logs,
and logging from a small real optimization, without running or comparing
full frontiers. Solver checks require the optional dependency and license:

```powershell
python -m unittest discover -s tests -v
```

### Machine used to generate the archived frontier

The author reports that the existing `data/pareto_of_capacities.csv` used
in the original experiments was generated on the following machine:

| Component | Specification |
|---|---|
| Device name | `DESKTOP-OJKJ4ND` |
| Processor | AMD Ryzen 3 2300U with Radeon Vega Mobile Gfx, 2.00 GHz |
| Installed memory | 12.0 GB RAM (10.9 GB usable) |
| Storage | 932 GB SSD, CT1000BX500SSD1 |
| Graphics | AMD Radeon Vega 6 Graphics (1010 MB reported) |
| System type | 64-bit operating system, x64-based processor |

These details describe the historical generation machine, not a required
configuration for running this package. The Python version, Gurobi version,
operating-system version, and any solver settings beyond those in the source
have not been supplied; the hardware information alone does not establish
them. Device and Windows product identifiers are omitted because they are
not needed to reproduce the computation.

### Persistent logs for new generation runs

`build_capacity_frontier()` now writes logs automatically. With the default
output filename, each invocation creates a unique, UTC-prefixed directory
under `data/pareto_regenerated_logs/`. Its location is printed to the console
as soon as the run starts. Repeated attempts use separate directories, so
logs from a failed attempt are not overwritten.

| File in the run directory | Contents |
|---|---|
| `progress.log` | Timestamped stages, environment versions, epsilon settings, layer bounds, solve start/end, statuses, objectives, accepted/duplicate points, export completion, and Python errors or interruptions |
| `minimum-anchor.log` | Native Gurobi progress for the minimum-capacity solve |
| `fairness-anchor.log` | Native Gurobi progress for the unbudgeted fairness solve |
| `sweep-0001.log`, etc. | Native Gurobi progress for each epsilon-constraint iteration |
| Corresponding `.prm` files | Nondefault Gurobi parameters recorded immediately before each solve; defaults depend on the logged solver version |

High-level progress is also printed to the console. Gurobi's detailed output
is kept in the per-solve files rather than flooding the console. Those logs
provide the solver's own progress information, including incumbent/bound/gap
and elapsed-time information when applicable. Python progress records are
flushed after each message, including before a potentially long solve.

To choose a different parent directory:

```python
frontier = build_capacity_frontier(
    load_networks(),
    filename="pareto_regenerated.csv",
    log_dir="run_logs",
)
```

To follow a run from PowerShell, substitute the directory printed at startup:

```powershell
Get-Content -LiteralPath '.\data\pareto_regenerated_logs\<run-directory>\progress.log' -Wait
```

Follow the active stage's `.log` file in another terminal for detailed solver
progress. The default generated-log directories are ignored by Git. Do not
commit logs without reviewing their contents: they can contain local paths
and solver/environment details.

These logs are diagnostic records, **not restart checkpoints**. Accepted
capacity vectors remain in memory until the final CSV export. A normal
Python exception or keyboard interruption is logged and re-raised; a hard
process kill, power loss, or native crash may leave no final error message.
The last flushed messages can still identify the stage that was active, but
cannot restore a lost solver state. Only `Run completed` confirms that the
final CSV was written and loaded successfully.

Logging takes effect only for newly started processes. Updating this code
does not attach logs to an already-running generation process, restart it,
or change its solver parameters. The original `data/pareto_of_capacities.csv`
remains read-only and is never used as a logging or checkpoint destination.

### Optional Gurobi 11+ logarithm handling

The `func_nonlinear` argument makes the logarithm representation explicit for
**both anchors and every epsilon-constraint solve**:

| Value | Behavior |
|---|---|
| `None` (default) | Do not override Gurobi's log-handling setting; retain the source-default path |
| `0` | Use Gurobi's static piecewise-linear approximation |
| `1` | Use nonlinear function constraints with dynamic outer approximation rather than a static log approximation |

Explicit selection requires Gurobi 11 or newer and sets Gurobi's
`FuncNonlinear` parameter. It does not change other numerical settings.

For an exact-nonlinear-log run:

```python
frontier = build_capacity_frontier(load_networks(), func_nonlinear=1)
```

The reason to expose this option is computational: the observed Gurobi 11.0.3
default-mode minimum-capacity solve expanded to 21,505 presolved columns and
1,475 SOS constraints. Its incumbent remained at a reported 0.12% gap after
an hour. Nonlinear mode avoids the static logarithm expansion, but is not a
guarantee that every instance will solve faster. These observations concern
the present regeneration run, not the machine that produced the archived
frontier.

**The relative MIP gap is not relaxed.** The implementation never sets
`MIPGap`; on the present Gurobi 11.0.3 installation its effective default is
`0.0001` (0.01%), not `0.01` (1%). A 1% tolerance was considered but was not
selected. The effective `FuncNonlinear` and `MIPGap` values are written to
`progress.log` before every solve, in addition to the existing Gurobi logs
and parameter snapshots.

“Exact nonlinear logs” describes the function representation, not
infinite-precision arithmetic or a zero optimality gap: solver feasibility
and optimality tolerances still apply. Selecting nonlinear rather than
piecewise-linear logs is an explicit run configuration, and can change the
returned capacity vector and subsequent layer bounds. No claim is made that
the archived experiment used this setting; its original solver logs are not
available. The original frontier and the reader's default selection remain
unchanged.

### Completed regeneration before the dependency correction: 26 October 2026

The full sweep completed in **80.47 seconds** with Python 3.8.0, Gurobi
11.0.3, `FuncNonlinear=1`, and the unchanged effective
`MIPGap=0.0001` (0.01%). The current machine reports an Intel Core Ultra 7
165H, 22 logical processors, and approximately 31.43 GB usable RAM. This is
a separate machine/configuration from the historical generation machine
listed above. The default-mode attempt was stopped; it did not produce a
completed frontier.

The archived frontier was generated approximately four years before
this rerun. The original solver logs and detailed software configuration
were not retained.
The rerun is therefore documented as a **configuration-specific regeneration,
not a replacement of the archived capacity artifact**.
The historical `data/pareto_of_capacities.csv` has not been modified.

Both files contain the same 360 directed capacity keys. The comparisons
below use sums of the exported capacity columns, not the historical
bookkeeping objective columns described above.

| Quantity | Archived frontier | Nonlinear regeneration |
|---|---:|---:|
| Exported solutions | 19 | 20 |
| Minimum-reference capacity sum | 10,048.78746149 | 10,048.78746155 |
| Minimum-reference Gas maximum | 420.700000 | 420.100000 |
| Minimum-reference Water maximum | 95.484000 | 115.600000 |
| Capacity sum at index 18 | 11,440.774734 | 12,029.259189 |
| Smallest capacity at index 18 | 12.849459 | 16.456843 |
| Capacity sum at the final exported point | 11,440.774734 | 10,051.497305 |

The minimum-reference sums differ by only about `6e-8`, while 154 individual
capacities differ by more than `1e-6` (maximum absolute difference:
`50.204`). This illustrates alternative minimum-capacity allocations with
effectively identical total capacity. Because their layer maxima become
the later logarithm bounds, different reference allocations can propagate
to different frontier samples.

At index 18, the regenerated total is **5.14% higher**, with a mean absolute
per-arc difference of `2.1853` and a maximum of `5.2934`. This comparison
aligns the export positions, not identical capacity budgets. The new run
also exports an additional point at index 19. Thus `row=-1` in the new file
does **not** represent the same choice as historical index 18; the existing
reader's historical default remains unchanged.

Solver version, logarithm representation, numerical settings, and the
chosen optimal reference vector are relevant run-specific factors. Without
the original logs, these observations cannot isolate the effect of hardware
alone or identify the original log mode. They do not by themselves establish
that the archived results are invalid.

#### Numerical verification of the exported capacities

A separate nominal-flow feasibility check, reading both files without
modifying either, accepted **all 19 archived solutions** at Gurobi's default
feasibility tolerance. It also accepted 19 of the 20 regenerated solutions.
Regenerated index 15 has a small numerical residual: its solver log reports
a maximum constraint violation of `7.9602e-6` and a maximum bound violation
of `2.7360e-6`. A diagnostic flow solve becomes feasible with a uniform
capacity allowance of approximately `2.39e-6` at the default `1e-6`
feasibility tolerance. This allowance was used only in the diagnostic
model; **no capacity file was adjusted**.

These checks concern nominal flow feasibility only, not disrupted-network
performance. An optimal-status flag alone does not establish that every
exported constraint is satisfied exactly.

### Regeneration with corrected dependencies: 26 October 2026

A subsequent run with the 21 dependency constraints, `FuncNonlinear=1`,
and unchanged solver tolerances completed in **343.83 seconds**, exporting
19 solutions to a separate file. Its minimum-reference capacity total is
10,048.78746155; its index-18 total is 12,001.465931, 4.90% above the
historical index-18 total and 0.23% below the earlier regeneration's
index-18 total. These are positional comparisons, not matched-budget optima.

A stronger recheck fixes each capacity vector in the full corrected nominal
model, including logarithms and reverse-direction capacity equalities, with
all functionality states set to one. The solver accepted 19/19 historical
solutions, 17/20 from the earlier regeneration, and 18/19 from the corrected
run. Corrected index 16 has a reverse-direction mismatch of approximately
`2.55e-6` and fails this recheck; the corrected run also reports native
constraint-violation warnings up to approximately `7.66e-6`. The new frontier
is therefore not claimed to be free of numerical residuals. No capacity
values were adjusted, and the historical frontier remains the default.
