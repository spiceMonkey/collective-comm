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

import pandas as pd


def sweep(
    curves: dict[str, Callable[[float], float]],
    x_values: Iterable[float],
    x_name: str = "x",
) -> pd.DataFrame:
    """Evaluate every curve at every ``x``; return long-form DataFrame.

    Each curve must return seconds; the result column ``t_us`` is in microseconds.
    """
    xs = list(x_values)
    rows = []
    for x in xs:
        for label, fn in curves.items():
            rows.append({x_name: x, "label": label, "t_us": fn(x) * 1e6})
    return pd.DataFrame(rows)


def log_space(start: float, stop: float, n: int) -> list[float]:
    """Log-spaced sequence (inclusive endpoints), no numpy dependency."""
    if n < 2:
        return [start]
    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / (n - 1)
    return [math.exp(log_start + i * step) for i in range(n)]
