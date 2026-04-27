"""Canonical anchor values for sweep notebooks.

Defaults mirror the regression anchors from
``documentation/modeling/04_in_network_collectives.md`` §3.1 and
``05_contention_and_congestion.md`` §6.1: N=512, M=16 MB, α=0.5 μs,
BW=900 GB/s. Override per-cell when sweeping.
"""

from dataclasses import dataclass, field


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

    # Hierarchical (NVL72-style scale-up + IB-class scale-out).
    L: int = 8                  # outer groups
    alpha_inner: float = 0.5e-6
    bw_inner: float = 900e9
    alpha_outer: float = 1.0e-6
    bw_outer: float = 50e9      # 400 Gbps NDR-class per-GPU bidi

    # Optional: total ranks for hierarchical helpers (defaults to G).
    N_hier: int = field(default=0)

    def __post_init__(self) -> None:
        if self.N_hier == 0:
            self.N_hier = self.G


# Contention profiles from 05_contention_and_congestion.md §4.1.
# Map: profile_name → (η_α, η_β).
ETA_PROFILES: dict[str, tuple[float, float]] = {
    "ideal":     (1.00, 1.00),
    "crossbar":  (1.00, 0.80),  # NVLink + NVSwitch, no SHARP
    "nvls":      (1.00, 0.52),  # NVLink SHARP in-network reduce
    "torus":     (1.20, 0.60),  # off-prefix concurrent groups
}


def to_us(t_seconds: float) -> float:
    """Seconds → microseconds."""
    return t_seconds * 1e6
