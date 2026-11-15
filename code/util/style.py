"""Shared chart style for the tutorial notebooks.

matplotlib is imported inside the functions, so importing ``util`` stays
stdlib-only. The colours are a validated categorical order (adjacent pairs
separate under simulated colour-vision deficiency). Three of them sit below
3:1 contrast on white, so every chart keeps a legend; exact values come from
the cost functions the notebooks call (most sweeps are also kept in DataFrames).
"""

import re

SURFACE = "#ffffff"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
RULE = "#c3c2b7"     # axis lines, connector rules
GRID = "#e6e5e1"     # hairline grid, one step off the surface

# Categorical slots, in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# One-hue ramp for an ordered quantity (light → dark).
SEQ_BLUE_5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# In charts that compare algorithms, colour follows the algorithm family rather
# than the position of a series, so "INC" or "torus" keeps its colour from one
# chart to the next. (Charts of phases or scenarios use SERIES in order.)
FAMILY_COLOR = {
    "ring": SERIES[0],          # ring / pairwise: linear-depth software
    "tree": SERIES[1],          # DBT, rec-doubling / halving, PAT, Bruck: log-depth software
    "inc": SERIES[2],           # in-network
    "rabenseifner": SERIES[3],
    "torus": SERIES[4],
}

_FAMILY_KEYWORDS = (
    ("inc", ("inc", "nvls", "hardware")),
    ("torus", ("torus",)),
    ("rabenseifner", ("rabenseifner",)),
    ("tree", ("dbt", "rec-doubling", "rec-halving", "pat", "bruck")),
    ("ring", ("ring", "pairwise")),
)


def family(label: str) -> str:
    """Algorithm family named by a curve label; raises on an unknown label."""
    words = set(re.findall(r"[a-z0-9\-]+", label.lower()))
    for name, keywords in _FAMILY_KEYWORDS:
        if words & set(keywords):
            return name
    raise ValueError(f"no algorithm family recognised in label {label!r}")


def color_for(label: str) -> str:
    return FAMILY_COLOR[family(label)]


def linestyle_for(label: str) -> str:
    """Secondary encoding for curves that share a colour or overlap another."""
    text = label.lower()
    if "non-pipelined" in text:
        return ":"      # same family colour as pipelined DBT
    if "rabenseifner" in text:
        return "--"     # lies on top of pipelined DBT
    return "-"


def tex(text: str) -> str:
    """Typeset η_α / η_β (and η_β_hw) with real subscripts in chart text."""
    return (text.replace("η_β_hw", r"$\eta_\beta^{\mathrm{hw}}$")
                .replace("η_β", r"$\eta_\beta$")
                .replace("η_α", r"$\eta_\alpha$"))


def apply() -> None:
    """Set matplotlib defaults: thin marks, hairline solid grid, quiet axes."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.figsize": (8.6, 4.6),
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.prop_cycle": cycler(color=SERIES),
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "lines.linewidth": 2.0,
        "lines.markersize": 7.5,
        "lines.markeredgecolor": SURFACE,
        "lines.markeredgewidth": 1.5,
        "lines.solid_capstyle": "round",
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.handlelength": 3.4,   # long enough to show a dashed or dotted line
    })


def legend_outside(ax, **kwargs):
    """Legend to the right of the axes, where it cannot cover the data."""
    return ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0.0, **kwargs)


def plain_log_ticks(axis, ticks) -> None:
    """Label a log axis at the given values in plain numbers (20, 50, 100, …)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())
