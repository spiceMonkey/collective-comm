"""Sweep helper — turn a dict of curve callables into a long-form DataFrame.

Notebooks declare curves like:

    curves = {
        "ring AR":  lambda M: ring_all_reduce(M, G, alpha, bw),
        "DBT AR":   lambda M: tree_all_reduce(M, G, alpha, bw, pipelined=True),
        "INC AR":   lambda M: inc_all_reduce(M, alpha, bw),
    }
    df = sweep(curves, log_space(1e3, 1e9, 60), x_name="M_bytes")

and get a DataFrame with columns ``[M_bytes, label, t_us]`` ready for
``df.pivot(...)`` or seaborn-style line plots.
"""

import math
from typing import Callable, Iterable


def sweep(
    curves: dict[str, Callable[[float], float]],
    x_values: Iterable[float],
    x_name: str = "x",
) -> "pd.DataFrame":
    """Evaluate every curve at every ``x``; return long-form DataFrame.

    Each curve must return seconds; the result column ``t_us`` is in microseconds.
    pandas is imported here, not at module import, so ``util`` (anchors,
    registry) stays usable with the standard library alone.
    """
    import pandas as pd

    if x_name in ("label", "t_us"):
        raise ValueError(f"x_name={x_name!r} would overwrite a result column")
    xs = list(x_values)
    rows = []
    for x in xs:
        for label, fn in curves.items():
            rows.append({x_name: x, "label": label, "t_us": fn(x) * 1e6})
    return pd.DataFrame(rows)


def log_space(start: float, stop: float, n: int) -> list[float]:
    """Log-spaced sequence of ``n`` points from ``start`` to ``stop`` (both
    included exactly), no numpy dependency. ``n = 1`` returns ``[start]``."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    if not start > 0 or not stop > 0:
        raise ValueError(f"log spacing needs positive endpoints, got {start!r} and {stop!r}")
    if n == 1:
        return [start]
    log_start = math.log(start)
    step = (math.log(stop) - log_start) / (n - 1)
    return [start] + [math.exp(log_start + i * step) for i in range(1, n - 1)] + [stop]
