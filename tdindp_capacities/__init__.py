"""Standalone reader for the tdINDP instance data and its arc capacities.

Quick start::

    from tdindp_capacities import load_networks, load_capacities

    nets = load_networks()
    u = load_capacities()          # row 18 of the corrected frontier
"""

from .capacities import (
    BENCHMARK_FILENAME,
    DEFAULT_ROW,
    FRONTIER_FILENAME,
    OBJECTIVE_COLUMNS,
    CapacityFrontier,
    compare_to_nominal,
    load_benchmark,
    load_capacities,
    load_frontier,
)
from .networks import DATA_DIR, NETWORKS, PRE_FIX_DATA_DIR, Gamma, Networks, load_networks
from .generation import build_capacity_frontier
from .moving_ceiling import build_moving_ceiling_frontier

__all__ = [
    "BENCHMARK_FILENAME",
    "DATA_DIR",
    "DEFAULT_ROW",
    "FRONTIER_FILENAME",
    "NETWORKS",
    "OBJECTIVE_COLUMNS",
    "PRE_FIX_DATA_DIR",
    "CapacityFrontier",
    "Gamma",
    "build_capacity_frontier",
    "build_moving_ceiling_frontier",
    "Networks",
    "compare_to_nominal",
    "load_benchmark",
    "load_capacities",
    "load_frontier",
    "load_networks",
]

__version__ = "0.1.0"
