"""Reading arc capacities off a pre-computed Pareto frontier.

Why this module exists
----------------------
The arc capacities ``u`` that the optimisation models actually use are
**not** the ``u`` column of ``GasArcs.csv`` / ``PowerArcs.csv`` /
``WaterArcs.csv``. Those nominal values are replaced wholesale by a row of a
frontier CSV. Not one of the 360 arcs keeps its CSV value.

Frontier files
--------------
* ``data/pareto_regenerated_2026-10-31.csv`` is the corrected frontier and the
  default. It is the output of ``generation.build_capacity_frontier``.
* ``pre_fix/data/benchmark_used_for_research_papers.csv`` is the frontier behind
  earlier publications. It is kept unchanged for traceability and read with
  :func:`load_benchmark`. Do not use it for new work.

The frontier
------------
The file has 19 rows and 362 columns: 360 arc capacities keyed by the literal
string ``"('Gas_node_9', 'Gas_node_0', 'Gas', 0)"``, plus the two objective
columns ``sum_u`` and ``sum_log_u``.

* **Row 0** is the *minimum-capacity* reference: total capacity is the smallest
  feasible (10,048.79) but capacity is distributed very unevenly -- the thinnest
  arc sits at exactly 1.0 unit and ``sum_log_u`` is only 789.62.
* **Rows 1-18** are the epsilon-constraint sweep. Row 1 maximises ``sum_log_u``
  at the fairness anchor's total (50,347.52); each subsequent row re-solves with
  ``sum_u <= previous * epsilon`` and ``epsilon`` shrinking by 0.99 per
  iteration. ``sum_u`` decreases monotonically, ``sum_log_u`` with it.
* **Row 18 is the last row**, the tightest point the sweep solved. Its total
  capacity (10,818.40) is within 8% of the minimum, yet ``sum_log_u`` is 1,056.96
  against 789.62 -- and its thinnest arc is 8.82 rather than 1.0.

Row 18 is a position, not a design: another frontier's row 18 is a different
vector. Prefer an explicit ``row`` or ``row=-1`` and record which file it came from.
"""

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .networks import DATA_DIR, PRE_FIX_DATA_DIR

#: The default selection: the last row of the shipped frontier.
DEFAULT_ROW = 18

#: Columns of the frontier CSV that are objectives, not arcs.
OBJECTIVE_COLUMNS = ("sum_log_u", "sum_u")

#: The corrected frontier, read by default.
FRONTIER_FILENAME = "pareto_regenerated_2026-10-31.csv"

#: The frontier behind earlier publications, ``pareto_of_capacities.csv`` in the
#: original code. It lives in ``pre_fix/data`` and is never modified.
BENCHMARK_FILENAME = "benchmark_used_for_research_papers.csv"

ArcTimeKey = Tuple[str, str, str, int]


@dataclass
class CapacityFrontier:
    """A sampled capacity design frontier."""

    table: pd.DataFrame  #: arc columns only
    objectives: pd.DataFrame  #: the file's raw ``sum_u`` / ``sum_log_u`` columns
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
        """Scalar diagnostics for one point, recomputed from its capacities."""
        caps = self.capacities(row)
        values = list(caps.values())
        return {
            "row": row if row >= 0 else self.n_points + row,
            "sum_u": sum(values),
            "sum_log_u": sum(math.log(v) for v in values),
            "n_arcs": len(values),
            "min_capacity": min(values),
            "max_capacity": max(values),
            "mean_capacity": sum(values) / len(values),
        }

    def summary_table(self) -> pd.DataFrame:
        """The whole frontier: objectives recomputed from the capacities, plus
        min/max capacity per point."""
        rows = []
        for r in range(self.n_points):
            values = list(self.capacities(r).values())
            rows.append({
                "row": r,
                "sum_u": sum(values),
                "sum_log_u": sum(math.log(v) for v in values),
                "min_capacity": min(values),
                "max_capacity": max(values),
                "role": _role(r, self.n_points),
            })
        return pd.DataFrame(rows)

    def objective_consistency(self) -> pd.DataFrame:
        """Check the stored ``sum_u`` / ``sum_log_u`` against the arc columns.

        They agree in the corrected frontier. They disagree systematically in
        the benchmark, because of the repeated arc in the pre-fix model -- see
        ``pre_fix/README.md``. :meth:`point_summary` and :meth:`summary_table`
        always report the recomputed values.
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
        return "minimum-capacity reference"
    if row == 1:
        return "fairness-anchor solution"
    if row == n_points - 1:
        return "terminal sweep point"
    return "epsilon sweep"


def load_frontier(data_dir: str = DATA_DIR, filename: str = FRONTIER_FILENAME) -> CapacityFrontier:
    """Read a frontier CSV (the corrected one by default) into a :class:`CapacityFrontier`.

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


def load_benchmark() -> CapacityFrontier:
    """Read the frontier behind earlier publications, for traceability only."""
    return load_frontier(data_dir=PRE_FIX_DATA_DIR, filename=BENCHMARK_FILENAME)


def load_capacities(row: int = DEFAULT_ROW, data_dir: str = DATA_DIR,
                    collapse_time: bool = False,
                    horizon: Optional[List[int]] = None) -> Dict[tuple, float]:
    """One-liner equivalent of the original ``find_capacities()``, on the
    corrected frontier."""
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
