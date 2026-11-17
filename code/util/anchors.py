"""Canonical anchor values for sweep notebooks.

Defaults mirror the regression anchors from
``documentation/modeling/04_in_network_collectives.md`` §3.1 and
``05_contention_and_congestion.md`` §5.1: N=512, M=16 MB, α=0.5 μs,
BW=900 GB/s. Override per-cell when sweeping.
"""

from dataclasses import dataclass


@dataclass
class Anchors:
    """Workload + topology parameters consumed by every primitive registry entry."""

    # Payload (bytes) and group rank count.
    M: float = 16e6
    G: int = 512

    # Single-tier (α, BW) — matches doc anchor (NVLink-class scale-up).
    alpha: float = 0.5e-6
    bw: float = 900e9

    # Torus / k-D mesh decomposition for the same N=512.
    dims: tuple = (8, 8, 8)

    # Hierarchical (NVLink-class inner tier + InfiniBand outer tier). The outer
    # values are 03_hierarchical_topologies.md §1.2's α_leaf ≈ 2 μs (α_spine is
    # ≈ 8 μs) and one 800 Gb/s XDR NIC — parameter values for the N = 512
    # sweeps, not a claim about which pods share a leaf switch.
    L: int = 8                  # outer groups
    alpha_inner: float = 0.5e-6
    bw_inner: float = 900e9
    alpha_outer: float = 2.0e-6
    bw_outer: float = 100e9     # 800 Gb/s XDR NIC = 100 GB/s per direction

    # Total ranks for the hierarchical helpers. 0 (the default) means "the same
    # as G"; read the resolved value through the ``n_hier`` property.
    N_hier: int = 0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_hier(self) -> int:
        """Rank count the hierarchical helpers use: ``N_hier``, or ``G`` when ``N_hier`` is 0."""
        return self.N_hier or self.G

    def validate(self) -> None:
        """Raise ``ValueError`` if the fields do not describe one operating point.

        ``dims`` must multiply out to ``G`` and ``L`` must divide ``n_hier``, so
        change them together with ``G`` (``Anchors(G=64, dims=(4, 4, 4))``).
        ``run_all`` calls this again, which covers fields changed after
        construction.
        """
        ranks = 1
        for d in self.dims:
            ranks *= d
        if ranks != self.G:
            raise ValueError(f"dims={self.dims} describe {ranks} ranks but G={self.G}; change them together")
        if self.L < 1 or self.n_hier % self.L != 0:
            raise ValueError(f"L={self.L} must divide the hierarchical rank count {self.n_hier}")


# Contention profiles from 05_contention_and_congestion.md §4.1.
# Map: profile_name → (η_α, η_β).
ETA_PROFILES: dict[str, tuple[float, float]] = {
    "ideal":     (1.00, 1.00),
    "crossbar":  (1.00, 0.80),  # NVLink + NVSwitch, no SHARP
    "nvls":      (1.00, 0.52),  # NVLink SHARP in-network reduce
    "torus":     (1.20, 0.60),  # assumed placeholder, not a measurement: concurrent groups sharing links, mild layout penalty
}


def to_us(t_seconds: float) -> float:
    """Seconds → microseconds."""
    return t_seconds * 1e6
