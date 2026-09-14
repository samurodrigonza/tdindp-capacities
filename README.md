# tdindp-capacities

A small, solver-free package that reads the tdINDP interdependent-network
instance (Gas / Power / Water) and — the point of the whole thing —
explains **where the arc capacities actually come from**.

It exists because that answer is genuinely surprising:

> The arc capacities used by the optimisation models are **not** the `u`
> column of `GasArcs.csv`, `PowerArcs.csv` or `WaterArcs.csv`. Those values
> are read, then discarded and replaced wholesale by a single row of a
> pre-computed Pareto frontier, `pareto_of_capacities.csv`.
> **Zero of the 360 arcs keep their CSV value.**

If you have ever written "arc capacities are taken from the network data
files" in a paper draft describing this instance, that sentence is wrong.

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
| `data/` | the instance CSVs plus `pareto_of_capacities.csv` |
| `example.py` | runnable, annotated walkthrough of everything below |

```bash
pip install -r requirements.txt
python example.py
```

Requires only `pandas`. No Gurobi, no license.

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

Not supported here — it needs Gurobi and the full tdINDP model. Set
`build = True` in the original `find_capacities()` if you must. If you do,
record the resulting row count and your selected row somewhere durable,
and switch the selection to `-1` or a named constant.
