"""Export the tutorial notebooks' figures as SVG for the mdBook.

    .venv/bin/python code/notebooks/export_figures.py

Re-runs the code cells of notebooks 01-04 (the same code the notebooks show)
with a non-interactive backend and writes each figure to
``documentation/modeling/assets/charts/``. Needs pandas and matplotlib, like
the notebooks themselves. Re-run it after changing a cost function or a
notebook so the figures in the book stay in step.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "documentation" / "modeling" / "assets" / "charts"

# One name per figure, in the order the notebook draws them.
FIGURES = {
    "01_payload_sweep.ipynb": [
        "ar_cost_vs_payload",
        "ar_goodput_vs_payload",
        "ag_cost_vs_payload",
    ],
    "02_group_topology_sweep.ipynb": [
        "ar_cost_vs_group_size",
        "a2a_cost_vs_group_size",
    ],
    "03_eta_sensitivity.ipynb": [
        "ar_ideal_vs_realistic",
        "ar_eta_beta_sensitivity",
        "ar_fat_tree_oversubscription",
    ],
    "04_hierarchical_partitioning.ipynb": [
        "hier_ar_phase_breakdown",
        "hier_ar_vs_outer_bw",
        "hier_ag_phase_order",
    ],
}


def export(nb_name: str, names: list[str]) -> None:
    nb = json.loads((HERE / nb_name).read_text())
    saved: list[str] = []

    def show(*_args, **_kwargs) -> None:
        for num in plt.get_fignums():
            name = names[len(saved)]
            plt.figure(num).savefig(
                OUT / f"{name}.svg", metadata={"Date": None}, bbox_inches="tight"
            )
            saved.append(name)
        plt.close("all")

    plt.show = show
    namespace = {"display": lambda *a, **k: None, "__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).split("\n")
        code = "\n".join(l for l in lines if not l.lstrip().startswith("%"))
        exec(compile(code, nb_name, "exec"), namespace)
    if saved != names:
        raise RuntimeError(f"{nb_name}: drew {len(saved)} figures, expected {len(names)}")
    print(f"{nb_name}: {', '.join(saved)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "collective_comm"  # reproducible element ids
    os.chdir(HERE)
    # Make `core` and `util` importable without relying on the notebooks' setup cells.
    sys.path.insert(0, str(HERE.parent))
    for nb_name, names in FIGURES.items():
        export(nb_name, names)


if __name__ == "__main__":
    main()
