"""Standalone reader for the tdINDP instance data and its arc capacities.

Quick start::

    from tdindp_capacities import load_networks, load_capacities

    nets = load_networks()
    u = load_capacities()          # frontier row 18, exactly as shipped
"""

from .capacities import (
    DEFAULT_ROW,
    FRONTIER_FILENAME,
    OBJECTIVE_COLUMNS,
    CapacityFrontier,
    compare_to_nominal,
    load_capacities,
    load_frontier,
)
from .networks import DATA_DIR, NETWORKS, Networks, load_networks

__all__ = [
    "DATA_DIR",
    "DEFAULT_ROW",
    "FRONTIER_FILENAME",
    "NETWORKS",
    "OBJECTIVE_COLUMNS",
    "CapacityFrontier",
    "Networks",
    "compare_to_nominal",
    "load_capacities",
    "load_frontier",
    "load_networks",
]

__version__ = "0.1.0"
