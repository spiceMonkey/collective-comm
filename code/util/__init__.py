"""Shared helpers for the collective_comm sweep notebooks.

Keep this layer thin — anchors and a primitive registry only. Cost equations
live in ``code/core/collective_cost.py`` and the docs under
``documentation/modeling/`` remain the source of truth.
"""

from .anchors import Anchors, ETA_PROFILES, to_us
from .registry import PRIMITIVES, PrimitiveSpec, run_all
from .sweep import sweep, log_space

__all__ = [
    "Anchors",
    "ETA_PROFILES",
    "to_us",
    "PRIMITIVES",
    "PrimitiveSpec",
    "run_all",
    "sweep",
    "log_space",
]
