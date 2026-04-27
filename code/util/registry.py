"""Registry of every primitive in ``core.collective_cost`` for bulk evaluation.

Each entry binds (group, name, doc_ref, callable) so a notebook can iterate the
full API at a fixed anchor in one pass — useful as a smoke test and as a quick
visual cross-section of which algorithms dominate at each operating point.
"""

from dataclasses import dataclass
from typing import Callable

from core import collective_cost as cc

from .anchors import Anchors


@dataclass(frozen=True)
class PrimitiveSpec:
    group: str                          # "BC", "Reduce", "AR", ...
    name: str                           # function name as registered (may include flag)
    doc_ref: str                        # full filename + section
    call: Callable[[Anchors], float]    # closure returning seconds
    notes: str = ""                     # short qualifier (e.g. "pipelined", "mesh")


def _bw_only_full(M: float, BW: float) -> str:
    """Throughput summary string for tabular display."""
    return f"{M / BW * 1e6:.2f} μs (M/BW)"


# ---------------------------------------------------------------------------
# Registry — one row per primitive at default flags. Pipelined / mesh variants
# are added as separate rows so the table reads as a wide cross-section.
# ---------------------------------------------------------------------------

PRIMITIVES: list[PrimitiveSpec] = [
    # --- Point-to-point ---------------------------------------------------
    PrimitiveSpec(
        "P2P", "p2p_hop", "01_collective_algorithms.md §8",
        lambda a: cc.p2p_hop(a.M, a.alpha, a.bw),
    ),

    # --- Broadcast --------------------------------------------------------
    PrimitiveSpec(
        "BC", "ring_broadcast", "01_collective_algorithms.md §3.1",
        lambda a: cc.ring_broadcast(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "BC", "ring_broadcast (pipelined)", "01_collective_algorithms.md §3.1 / App. C",
        lambda a: cc.ring_broadcast(a.M, a.G, a.alpha, a.bw, pipelined=True),
        notes="P→P*",
    ),
    PrimitiveSpec(
        "BC", "tree_broadcast", "01_collective_algorithms.md §3.2",
        lambda a: cc.tree_broadcast(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "BC", "tree_broadcast (pipelined)", "01_collective_algorithms.md §3.2 / App. C",
        lambda a: cc.tree_broadcast(a.M, a.G, a.alpha, a.bw, pipelined=True),
        notes="P→P*",
    ),
    PrimitiveSpec(
        "BC", "inc_broadcast", "04_in_network_collectives.md §1.2",
        lambda a: cc.inc_broadcast(a.M, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "BC", "torus_broadcast", "02_topology_mapping.md §3.2",
        lambda a: cc.torus_broadcast(a.M, a.dims, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "BC", "torus_broadcast (mesh)", "02_topology_mapping.md §4.2",
        lambda a: cc.torus_broadcast(a.M, a.dims, a.alpha, a.bw, wraparound=False),
        notes="open-line per dim",
    ),

    # --- Reduce -----------------------------------------------------------
    PrimitiveSpec(
        "Reduce", "ring_reduce", "01_collective_algorithms.md §4.1",
        lambda a: cc.ring_reduce(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "Reduce", "tree_reduce", "01_collective_algorithms.md §4.2",
        lambda a: cc.tree_reduce(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "Reduce", "inc_reduce", "04_in_network_collectives.md §1.1",
        lambda a: cc.inc_reduce(a.M, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "Reduce", "torus_reduce", "02_topology_mapping.md §3.3",
        lambda a: cc.torus_reduce(a.M, a.dims, a.alpha, a.bw),
    ),

    # --- All-reduce -------------------------------------------------------
    PrimitiveSpec(
        "AR", "ring_all_reduce", "01_collective_algorithms.md §5.1",
        lambda a: cc.ring_all_reduce(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AR", "tree_all_reduce", "01_collective_algorithms.md §5.2",
        lambda a: cc.tree_all_reduce(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AR", "tree_all_reduce (pipelined)", "01_collective_algorithms.md §5.2 / App. C",
        lambda a: cc.tree_all_reduce(a.M, a.G, a.alpha, a.bw, pipelined=True),
        notes="P→P*",
    ),
    PrimitiveSpec(
        "AR", "rabenseifner_all_reduce", "01_collective_algorithms.md App. B.2",
        lambda a: cc.rabenseifner_all_reduce(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AR", "inc_all_reduce", "04_in_network_collectives.md §1.4 / §2",
        lambda a: cc.inc_all_reduce(a.M, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AR", "torus_all_reduce", "02_topology_mapping.md §3.4",
        lambda a: cc.torus_all_reduce(a.M, a.dims, a.alpha, a.bw),
    ),

    # --- All-gather -------------------------------------------------------
    PrimitiveSpec(
        "AG", "ring_all_gather", "01_collective_algorithms.md §6",
        lambda a: cc.ring_all_gather(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AG", "recursive_doubling_all_gather", "01_collective_algorithms.md App. B.4",
        lambda a: cc.recursive_doubling_all_gather(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AG", "pat_all_gather", "01_collective_algorithms.md App. A",
        lambda a: cc.pat_all_gather(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AG", "inc_all_gather", "04_in_network_collectives.md §1.4",
        lambda a: cc.inc_all_gather(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "AG", "torus_all_gather", "02_topology_mapping.md §3.5",
        lambda a: cc.torus_all_gather(a.M, a.dims, a.alpha, a.bw),
    ),

    # --- Reduce-scatter ---------------------------------------------------
    PrimitiveSpec(
        "RS", "ring_reduce_scatter", "01_collective_algorithms.md §6",
        lambda a: cc.ring_reduce_scatter(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "RS", "recursive_halving_reduce_scatter", "01_collective_algorithms.md App. B.4",
        lambda a: cc.recursive_halving_reduce_scatter(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "RS", "pat_reduce_scatter", "01_collective_algorithms.md App. A",
        lambda a: cc.pat_reduce_scatter(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "RS", "inc_reduce_scatter", "04_in_network_collectives.md §1.4",
        lambda a: cc.inc_reduce_scatter(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "RS", "torus_reduce_scatter", "02_topology_mapping.md §3.5",
        lambda a: cc.torus_reduce_scatter(a.M, a.dims, a.alpha, a.bw),
    ),

    # --- All-to-all -------------------------------------------------------
    PrimitiveSpec(
        "A2A", "pairwise_a2a", "01_collective_algorithms.md §7.2",
        lambda a: cc.pairwise_a2a(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "A2A", "ring_relay_a2a", "01_collective_algorithms.md §7.1",
        lambda a: cc.ring_relay_a2a(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "A2A", "bruck_a2a", "01_collective_algorithms.md App. B.5",
        lambda a: cc.bruck_a2a(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "A2A", "inc_a2a", "04_in_network_collectives.md §1.3",
        lambda a: cc.inc_a2a(a.M, a.G, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "A2A", "torus_a2a", "02_topology_mapping.md §3.6",
        lambda a: cc.torus_a2a(a.M, a.dims, a.alpha, a.bw),
    ),
    PrimitiveSpec(
        "A2A", "torus_a2a (mesh)", "02_topology_mapping.md §4.2",
        lambda a: cc.torus_a2a(a.M, a.dims, a.alpha, a.bw, wraparound=False),
        notes="open-line per dim",
    ),

    # --- Hierarchical -----------------------------------------------------
    PrimitiveSpec(
        "Hierarchical", "hierarchical_all_reduce_ring_ring",
        "03_hierarchical_topologies.md §2.1",
        lambda a: cc.hierarchical_all_reduce_ring_ring(
            a.M, a.N_hier, a.L,
            a.alpha_inner, a.bw_inner,
            a.alpha_outer, a.bw_outer,
        ),
        notes="ring inner + ring outer",
    ),
]


def run_all(anchors: Anchors | None = None) -> "list[dict]":
    """Evaluate every entry in ``PRIMITIVES`` at the given anchor and return rows.

    Each row: ``{group, name, doc_ref, notes, t_us}``. No external deps —
    callers can wrap in ``pandas.DataFrame(...)`` or print directly.
    """
    a = anchors or Anchors()
    rows = []
    for spec in PRIMITIVES:
        t_s = spec.call(a)
        rows.append({
            "group": spec.group,
            "name": spec.name,
            "doc_ref": spec.doc_ref,
            "notes": spec.notes,
            "t_us": t_s * 1e6,
        })
    return rows
