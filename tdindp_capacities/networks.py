"""Standalone loader for the tdINDP interdependent-network instance.

This is a self-contained rewrite of the data-reading half of
``tdINDP_parameters_build.load_tdINDP_parameters``. It has no solver
dependency: it only reads CSVs and builds the index sets and parameter
dictionaries.

Naming follows the paper:

====================  =========================================================
``K``                 list of networks (layers), e.g. ``['Gas','Power','Water']``
``N_k[k]``            nodes of network ``k``, as ``'<Network>_node_<id>'``
``A_k[k]``            directed arcs of network ``k``, as ``(tail, head)``
``b[(i,k,l,t)]``      supply (>0) / demand (<0) of commodity ``l`` at node ``i``
``u[(i,j,k,t)]``      capacity of arc ``(i,j)`` -- see ``capacities.py``
``c[(i,j,k,l,t)]``    unit flow cost
``f[(i,j,k,t)]``      arc reconstruction cost
``q[(i,k,t)]``        node reconstruction cost
``M_plus/M_minus``    penalties for unused supply / unmet demand
``gamma[(i,j,k,kt,t)]``  node ``i`` in ``k`` is a *dependee* of node ``j`` in ``kt``
====================  =========================================================

Three behaviours are easy to trip over, so they are called out explicitly:

1. **Arcs are made bidirectional after loading.** Each unique CSV arc yields
   one directed arc; the reverse arc is then appended if absent, and ``u``,
   ``f`` and ``c`` are mirrored onto it. The arc counts therefore double
   relative to the unique CSV arcs.
2. **``gamma`` keys are 5-tuples ending in ``t``.** A 4-tuple lookup in the
   original code silently returned nothing, which once disabled every
   interdependency constraint in the downstream models. :class:`Gamma` now
   raises on any lookup whose key is not a 5-tuple.
3. **``PowerArcs.csv`` lists arc 5 -> 64 on two rows.** ``A_k`` keeps it once.
   As in the original dictionaries, ``f`` and ``c`` take the later row's values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

#: Data behind earlier publications; see ``pre_fix/README.md``.
PRE_FIX_DATA_DIR = os.path.join(_ROOT, "pre_fix", "data")

#: Networks actually loaded. ``Telecommunication`` CSVs ship with the data
#: but are excluded here, exactly as in the original study.
NETWORKS = ["Gas", "Power", "Water"]

NodeKey = str
ArcKey = Tuple[str, str]


class Gamma(dict):
    """Interdependency map that rejects lookups with a malformed key.

    Keys are ``(dependee, depender, dependee_network, depender_network, t)``.
    Raises ``ValueError`` rather than ``KeyError`` so that a wrong key shape
    cannot be mistaken for an absent dependency.
    """

    @staticmethod
    def _check(key) -> None:
        if not (isinstance(key, tuple) and len(key) == 5):
            raise ValueError(
                "gamma keys are (dependee, depender, dependee_network, "
                "depender_network, t); got {!r}".format(key)
            )

    def __contains__(self, key) -> bool:
        self._check(key)
        return super().__contains__(key)

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)


@dataclass
class Networks:
    """The loaded instance."""

    K: List[str]
    T: List[int]
    N_k: Dict[str, List[NodeKey]]
    A_k: Dict[str, List[ArcKey]]
    L_k: Dict[str, List[str]]
    b: Dict[tuple, float]
    u_nominal: Dict[tuple, float]
    c: Dict[tuple, float]
    f: Dict[tuple, float]
    q: Dict[tuple, float]
    M_plus: Dict[tuple, float] = field(default_factory=dict)
    M_minus: Dict[tuple, float] = field(default_factory=dict)
    gamma: Dict[tuple, int] = field(default_factory=Gamma)

    # -- node partitions ---------------------------------------------------
    def supply_nodes(self, k: str, t: int = 0) -> List[NodeKey]:
        """Nodes of ``k`` with positive net availability."""
        return [i for i in self.N_k[k] if self.b[(i, k, k, t)] > 0]

    def demand_nodes(self, k: str, t: int = 0) -> List[NodeKey]:
        """Nodes of ``k`` with negative net availability."""
        return [i for i in self.N_k[k] if self.b[(i, k, k, t)] < 0]

    def transshipment_nodes(self, k: str, t: int = 0) -> List[NodeKey]:
        """Nodes of ``k`` that neither supply nor demand."""
        return [i for i in self.N_k[k] if self.b[(i, k, k, t)] == 0]

    def arc_keys(self, t: int = 0) -> List[tuple]:
        """Every ``(i, j, k, t)`` capacity key, in network order."""
        return [(i, j, k, t) for k in self.K for (i, j) in self.A_k[k]]

    def total_supply(self, t: int = 0) -> Dict[str, float]:
        return {k: sum(self.b[(i, k, k, t)] for i in self.supply_nodes(k, t)) for k in self.K}

    def total_demand(self, t: int = 0) -> Dict[str, float]:
        return {k: -sum(self.b[(i, k, k, t)] for i in self.demand_nodes(k, t)) for k in self.K}

    def summary(self) -> pd.DataFrame:
        """One row per network: node/arc counts and supply/demand totals."""
        sup, dem = self.total_supply(), self.total_demand()
        return pd.DataFrame(
            [
                {
                    "network": k,
                    "nodes": len(self.N_k[k]),
                    "arcs": len(self.A_k[k]),
                    "supply_nodes": len(self.supply_nodes(k)),
                    "demand_nodes": len(self.demand_nodes(k)),
                    "total_supply": sup[k],
                    "total_demand": dem[k],
                }
                for k in self.K
            ]
        )


def _prefixed(network: str, node_id) -> str:
    return "{}_node_{}".format(network, node_id)


def load_networks(data_dir: str = DATA_DIR, networks: List[str] = None,
                  horizon: List[int] = None) -> Networks:
    """Read the node/arc/interdependency CSVs into index sets and parameters.

    Parameters
    ----------
    data_dir:
        Folder holding ``<Network>Nodes.csv``, ``<Network>Arcs.csv`` and
        ``Interdep.csv``.
    networks:
        Subset of :data:`NETWORKS` to load. Defaults to all three.
    horizon:
        Time periods to replicate parameters over. The published study is
        single-period, so this defaults to ``[0]``.
    """
    K = list(NETWORKS if networks is None else networks)
    T = [0] if horizon is None else list(horizon)

    N_k: Dict[str, List[str]] = {}
    A_k: Dict[str, List[ArcKey]] = {}
    L_k: Dict[str, List[str]] = {k: [k] for k in K}
    b, u, c, f, q, M_plus, M_minus, gamma = {}, {}, {}, {}, {}, {}, {}, {}

    for k in K:
        nodes = pd.read_csv(os.path.join(data_dir, "{}Nodes.csv".format(k)))
        N_k[k] = [_prefixed(k, i) for i in nodes["ID"]]
        for t in T:
            for row in nodes.index:
                i = _prefixed(k, nodes["ID"][row])
                b[(i, k, k, t)] = float(nodes["Demand"][row])
                q[(i, k, t)] = float(nodes["q (complete DS)"][row])
                M_plus[(i, k, k, t)] = float(nodes["Mp"][row])
                M_minus[(i, k, k, t)] = float(nodes["Mm"][row])

        arcs = pd.read_csv(os.path.join(data_dir, "{}Arcs.csv".format(k)))
        # dict.fromkeys drops repeated CSV rows but keeps the file order.
        A_k[k] = list(dict.fromkeys(
            (_prefixed(k, arcs["Start Node"][r]), _prefixed(k, arcs["End Node"][r]))
            for r in arcs.index
        ))
        for t in T:
            for r in arcs.index:
                i = _prefixed(k, arcs["Start Node"][r])
                j = _prefixed(k, arcs["End Node"][r])
                u[(i, j, k, t)] = float(arcs["u"][r])
                f[(i, j, k, t)] = float(arcs["f"][r])
                c[(i, j, k, k, t)] = float(arcs["c"][r])

    # Interdependencies. NOTE the 5-tuple key: (dependee, depender,
    # dependee_network, depender_network, t).
    interdep = pd.read_csv(os.path.join(data_dir, "Interdep.csv"))
    for t in T:
        for row in interdep.index:
            dependee_net = interdep["Dependee Network"][row]
            depender_net = interdep["Depender Network"][row]
            if dependee_net in K and depender_net in K:
                key = (
                    _prefixed(dependee_net, interdep["Dependee Node"][row]),
                    _prefixed(depender_net, interdep["Depender Node"][row]),
                    dependee_net,
                    depender_net,
                    t,
                )
                gamma[key] = 1

    _make_bidirectional(K, T, A_k, L_k, u, f, c)

    return Networks(K=K, T=T, N_k=N_k, A_k=A_k, L_k=L_k, b=b, u_nominal=u,
                    c=c, f=f, q=q, M_plus=M_plus, M_minus=M_minus, gamma=Gamma(gamma))


def _make_bidirectional(K, T, A_k, L_k, u, f, c) -> None:
    """Append every missing reverse arc and mirror ``u``, ``f`` and ``c``.

    This is what takes the instance from 180 unique CSV arcs to 360, and it
    is why the frontier CSVs have 360 capacity columns.
    """
    for k in K:
        for (i, j) in list(A_k[k]):
            if (j, i) not in A_k[k]:
                A_k[k].append((j, i))
    for k in K:
        for (i, j) in A_k[k]:
            for t in T:
                if (i, j, k, t) in f and (j, i, k, t) not in f:
                    f[(j, i, k, t)] = f[(i, j, k, t)]
                if (i, j, k, t) in u and (j, i, k, t) not in u:
                    u[(j, i, k, t)] = u[(i, j, k, t)]
                for l in L_k[k]:
                    if (i, j, k, l, t) in c and (j, i, k, l, t) not in c:
                        c[(j, i, k, l, t)] = c[(i, j, k, l, t)]
