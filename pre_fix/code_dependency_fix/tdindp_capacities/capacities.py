"""Reading arc capacities off the pre-computed Pareto frontier.

Why this module exists
----------------------
The arc capacities ``u`` that the optimisation models actually use are
**not** the ``u`` column of ``GasArcs.csv`` / ``PowerArcs.csv`` /
``WaterArcs.csv``. Those nominal values are read, then thrown away and
replaced wholesale by a row of ``pareto_of_capacities.csv``. Not one of
the 360 arcs keeps its CSV value.

In the original code this happens inside ``find_capacities()``, gated by a
hard-coded ``build`` flag:

* ``build = True``  -- solve the bi-objective capacity design problem and
  *write* ``pareto_of_capacities.csv``. Requires Gurobi. Slow.
* ``build = False`` -- (the shipped setting) *read* the CSV back and select
  ``row_number = 18``.

This module reproduces the ``build = False`` path exactly, with no solver
dependency, and makes the row choice explicit instead of hard-coded.

The frontier
------------
``pareto_of_capacities.csv`` has 19 rows and 362 columns: 360 arc
capacities keyed by the literal string ``"('Gas_node_9', 'Gas_node_0',
'Gas', 0)"``, plus the two objective columns ``sum_u`` and ``sum_log_u``.

* **Row 0** is the *cost-optimal* reference design: total capacity is the
  cheapest feasible (``sum_u`` = 10,060.76) but capacity is distributed
  very unevenly -- the thinnest arc sits at exactly 1.0 unit and
  ``sum_log_u`` is only 790.43.
* **Rows 1-18** are the epsilon-constraint sweep. Row 1 maximises
  ``sum_log_u`` with no budget cap (``sum_u`` = 50,889.06); each subsequent
  row re-solves with ``sum_u <= previous * epsilon`` and ``epsilon``
  shrinking by 0.99 per iteration. ``sum_u`` decreases monotonically,
  ``sum_log_u`` with it.
* **Row 18 is the last row**, i.e. ``df.iloc[-1]``. It is the tightest
  point the sweep still solved. Its total capacity (11,440.77) is within
  14% of the cost-optimal design, yet ``sum_log_u`` is 1,100.41 against
  790.43 -- and its thinnest arc is 12.85 rather than 1.0.

So ``row_number = 18`` is not an arbitrary pick: it is "as close to the
cost-optimal capacity budget as the sweep got, but spread proportionally
fairly instead of concentrated". That said, it *is* recorded nowhere in
the run provenance, and it is written as a literal ``18`` rather than
``-1``, so regenerating the frontier with a different number of sweep
iterations would silently change which design gets used. Prefer
:func:`load_capacities` with an explicit ``row`` or ``row=-1``.
"""

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .networks import DATA_DIR

#: The row the original ``find_capacities()`` hard-codes. Also the last row.
DEFAULT_ROW = 18

#: Columns of the frontier CSV that are objectives, not arcs.
OBJECTIVE_COLUMNS = ("sum_log_u", "sum_u")

FRONTIER_FILENAME = "pareto_of_capacities.csv"

ArcTimeKey = Tuple[str, str, str, int]


@dataclass
class CapacityFrontier:
    """The 19-point capacity design frontier."""

    table: pd.DataFrame  #: arc columns only, index 0..18
    objectives: pd.DataFrame  #: ``sum_u`` / ``sum_log_u``, index 0..18
    arc_keys: List[ArcTimeKey]
    path: str

    @property
    def n_points(self) -> int:
        return len(self.table)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_keys)

    def capacities(self, row: int = DEFAULT_ROW, collapse_time: bool = False,
                   horizon: Optional[List[int]] = None) -> Dict[tuple, float]:
        """Return ``{(i, j, k, t): capacity}`` for one frontier point.

        Parameters
        ----------
        row:
            Frontier index. ``-1`` selects the last point, which for the
            shipped CSV is the same design as :data:`DEFAULT_ROW`.
        collapse_time:
            If ``True``, drop the trailing ``t`` and key by ``(i, j, k)``.
            The downstream single-period models do this immediately after
            loading.
        horizon:
            Replicate the ``t = 0`` capacities across these periods. The
            original code does the same for multi-period runs, since the
            frontier only ever stores ``t = 0``.
        """
        series = self.table.iloc[row]
        caps = {ast.literal_eval(col): float(val) for col, val in series.items()}

        if horizon:
            for (i, j, k, _t) in list(caps):
                for t in horizon:
                    if t != 0:
                        caps[(i, j, k, t)] = caps[(i, j, k, 0)]

        if collapse_time:
            return {(i, j, k): v for (i, j, k, _t), v in caps.items()}
        return caps

    def point_summary(self, row: int = DEFAULT_ROW) -> Dict[str, float]:
        """Scalar diagnostics for one frontier point."""
        caps = self.capacities(row)
        values = list(caps.values())
        return {
            "row": row if row >= 0 else self.n_points + row,
            "sum_u": float(self.objectives["sum_u"].iloc[row]),
            "sum_log_u": float(self.objectives["sum_log_u"].iloc[row]),
            "n_arcs": len(values),
            "min_capacity": min(values),
            "max_capacity": max(values),
            "mean_capacity": sum(values) / len(values),
            "recomputed_sum_log_u": sum(math.log(v) for v in values),
        }

    def summary_table(self) -> pd.DataFrame:
        """The whole frontier: objectives plus min/max capacity per point."""
        rows = []
        for r in range(self.n_points):
            values = list(self.capacities(r).values())
            rows.append({
                "row": r,
                "sum_u": float(self.objectives["sum_u"].iloc[r]),
                "sum_log_u": float(self.objectives["sum_log_u"].iloc[r]),
                "min_capacity": min(values),
                "max_capacity": max(values),
                "role": _role(r, self.n_points),
            })
        return pd.DataFrame(rows)

    def objective_consistency(self) -> pd.DataFrame:
        """Check the stored ``sum_u`` / ``sum_log_u`` against the arc columns.

        They do not fully agree, and the disagreement is systematic --
        see the "Known data quirks" section of the README. Use this to
        confirm the quirk is still present in whatever copy of the CSV you
        are holding.
        """
        rows = []
        for r in range(self.n_points):
            values = list(self.capacities(r).values())
            stored_u = float(self.objectives["sum_u"].iloc[r])
            stored_log = float(self.objectives["sum_log_u"].iloc[r])
            recomputed_u = sum(values)
            recomputed_log = sum(math.log(v) for v in values)
            rows.append({
                "row": r,
                "stored_sum_u": stored_u,
                "recomputed_sum_u": recomputed_u,
                "sum_u_error": recomputed_u - stored_u,
                "stored_sum_log_u": stored_log,
                "recomputed_sum_log_u": recomputed_log,
                "sum_log_u_error": recomputed_log - stored_log,
            })
        return pd.DataFrame(rows)

    def as_frame(self, row: int = DEFAULT_ROW) -> pd.DataFrame:
        """One frontier point as a tidy ``tail/head/network/capacity`` table."""
        caps = self.capacities(row)
        return pd.DataFrame(
            [{"tail": i, "head": j, "network": k, "t": t, "capacity": v}
             for (i, j, k, t), v in caps.items()]
        )

    def compare(self, row_a: int, row_b: int) -> pd.DataFrame:
        """Per-arc capacity difference between two frontier points."""
        a, b = self.capacities(row_a), self.capacities(row_b)
        return pd.DataFrame(
            [{"tail": i, "head": j, "network": k,
              "capacity_a": a[(i, j, k, t)], "capacity_b": b[(i, j, k, t)],
              "delta": b[(i, j, k, t)] - a[(i, j, k, t)]}
             for (i, j, k, t) in a]
        ).sort_values("delta")


def _role(row: int, n_points: int) -> str:
    if row == 0:
        return "cost-optimal reference"
    if row == 1:
        return "unconstrained max sum_log_u"
    if row == n_points - 1:
        return "tightest sweep point (the one used)"
    return "epsilon sweep"


def load_frontier(data_dir: str = DATA_DIR, filename: str = FRONTIER_FILENAME) -> CapacityFrontier:
    """Read ``pareto_of_capacities.csv`` into a :class:`CapacityFrontier`.

    The first column of the file is an unnamed pandas index artefact and is
    dropped, mirroring ``df.drop(df.columns[0], axis=1)`` in the original.
    """
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)

    objectives = df[list(OBJECTIVE_COLUMNS)].copy()
    table = df.drop(columns=list(OBJECTIVE_COLUMNS))
    arc_keys = [ast.literal_eval(col) for col in table.columns]

    return CapacityFrontier(table=table, objectives=objectives,
                            arc_keys=arc_keys, path=path)


def load_capacities(row: int = DEFAULT_ROW, data_dir: str = DATA_DIR,
                    collapse_time: bool = False,
                    horizon: Optional[List[int]] = None) -> Dict[tuple, float]:
    """One-liner equivalent of the original ``find_capacities()``.

    ``load_capacities()`` with no arguments reproduces the shipped
    behaviour bit for bit.
    """
    return load_frontier(data_dir).capacities(row=row, collapse_time=collapse_time,
                                              horizon=horizon)


def compare_to_nominal(networks, row: int = DEFAULT_ROW,
                       data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Frontier capacities against the ``u`` column of the arc CSVs.

    Every arc differs, which is the single most important thing to know
    about this dataset: the CSV capacities are decorative.
    """
    frontier = load_capacities(row=row, data_dir=data_dir)
    rows = []
    for key, value in frontier.items():
        i, j, k, t = key
        nominal = networks.u_nominal.get(key)
        rows.append({
            "tail": i, "head": j, "network": k,
            "nominal_csv": nominal,
            "frontier": value,
            "delta": None if nominal is None else value - nominal,
        })
    return pd.DataFrame(rows)
