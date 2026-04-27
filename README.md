# collective_comm

A first-principles framework for analyzing **collective-communication trade-offs** in distributed AI clusters: α-β cost models, topology-aware algorithms, in-network collectives, and contention coefficients. Designed as a **tutorial + cost-analysis toolkit** — not a runtime library.

If you've ever wondered:
- *Why does NCCL pick DBT for small messages but ring for large ones?*
- *What does NVLS / SHARP actually buy you over a software all-reduce?*
- *How should I split TP/EP/PP/DP across NVL72 vs InfiniBand tiers?*
- *How much performance does fat-tree oversubscription cost in practice?*

…this repo gives you the equations, calibrations, and runnable notebooks to answer them.

---

## Layout

```
collective_comm/
├── documentation/modeling/   ← the spec — every cost formula derives here
│   ├── 00_summary.md         ← entry point / cheatsheet
│   ├── 01_collective_algorithms.md
│   ├── 02_topology_mapping.md
│   ├── 03_hierarchical_topologies.md
│   ├── 04_in_network_collectives.md
│   ├── 05_contention_and_congestion.md
│   └── references.md
└── code/
    ├── core/collective_cost.py   ← α-β cost primitives (BC, Reduce, AR, AG/RS, A2A)
    ├── util/                     ← anchors, primitive registry, sweep helpers
    └── notebooks/
        ├── 00_api_enumeration.ipynb           ← every primitive at the canonical anchor
        ├── 01_payload_sweep.ipynb             ← M from 1 KB → 1 GB; α/BW regime crossover
        ├── 02_group_topology_sweep.ipynb      ← G scaling; torus dimensionality
        ├── 03_eta_sensitivity.ipynb           ← realistic η contention; fat-tree oversub
        └── 04_hierarchical_partitioning.ipynb ← two-tier composition; NVL72 + IB sweep
```

---

## Modeling docs (the source of truth)

The series under `documentation/modeling/` is the contract the code implements. Read it before editing any cost function — start with `documentation/modeling/00_summary.md` for the cheatsheet view.

| Doc | Covers |
|---|---|
| `01_collective_algorithms.md` | α-β model; ring / DBT / Rabenseifner / PAT / Bruck per primitive |
| `02_topology_mapping.md` | Single-tier specializations: star, k-D torus, k-D mesh |
| `03_hierarchical_topologies.md` | Multi-tier Clos / fat-tree composition; TP/EP/PP/DP allocation |
| `04_in_network_collectives.md` | NVLS / SHARP / Tomahawk Ultra; switch-ALU + multicast |
| `05_contention_and_congestion.md` | η_α, η_β coefficients; calibration profiles |
| `references.md` | Verified equations and primary sources |

---

## Quick start

```bash
# One-time setup (Python 3.14)
python3.14 -m venv .venv
.venv/bin/pip install jupyterlab pandas matplotlib ipykernel

# Open the tutorial notebooks
.venv/bin/jupyter lab code/notebooks/
```

The core cost functions are stdlib-only — you only need pandas / matplotlib for the notebooks.

---

## Examples

### 1. Compare AR algorithms at the canonical anchor

```python
from core.collective_cost import (
    ring_all_reduce, tree_all_reduce, rabenseifner_all_reduce,
    inc_all_reduce, torus_all_reduce,
)

# N=512 ranks, 16 MB AR payload, NVLink-class fabric
M, G, alpha, bw = 16e6, 512, 0.5e-6, 900e9

print(f"ring AR:           {ring_all_reduce(M, G, alpha, bw) * 1e6:7.1f} μs")  # 546.5
print(f"DBT AR:            {tree_all_reduce(M, G, alpha, bw) * 1e6:7.1f} μs")  # 169.0
print(f"DBT AR pipelined:  {tree_all_reduce(M, G, alpha, bw, pipelined=True) * 1e6:7.1f} μs")  # 26.8
print(f"Rabenseifner AR:   {rabenseifner_all_reduce(M, G, alpha, bw) * 1e6:7.1f} μs")  # 44.5
print(f"INC AR (NVLS):     {inc_all_reduce(M, alpha, bw) * 1e6:7.1f} μs")      # 18.8
print(f"torus 8³ AR:       {torus_all_reduce(M, (8,8,8), alpha, bw) * 1e6:7.1f} μs")  # 56.5
```

### 2. Apply realistic η contention

```python
from core.collective_cost import inc_all_reduce, apply_eta

# NVLS calibration from 05_contention_and_congestion.md §4.1
alpha_eff, bw_eff = apply_eta(alpha, bw, eta_alpha=1.00, eta_beta=0.52)
print(f"INC AR (realistic): {inc_all_reduce(M, alpha_eff, bw_eff) * 1e6:.1f} μs")  # ~35
```

### 3. Hierarchical AR for an NVL72 + IB cluster

For 576 GPUs split as **8 racks × 72 GPUs/rack** (`L=8`, inner = NVLink, outer = IB):

```python
from core.collective_cost import hierarchical_all_reduce_ring_ring

t = hierarchical_all_reduce_ring_ring(
    M=16e6, N=576, L=8,                  # 8 outer groups, 72 inner each
    alpha_inner=0.5e-6, bw_inner=900e9,  # NVLink-class
    alpha_outer=1.0e-6, bw_outer=50e9,   # 400 Gbps IB-class
)
print(f"hierarchical AR: {t * 1e6:.1f} μs")
```

The outer sub-AR ships only the *telescoped* payload `M·L/N` — see `documentation/modeling/03_hierarchical_topologies.md` §2.1 for the derivation.

### 4. Sweep across η profiles using the registry

```python
from util import Anchors, ETA_PROFILES, run_all
from core.collective_cost import apply_eta

a = Anchors()  # canonical N=512 anchor
for name, row in {r["name"]: r for r in run_all(a)}.items():
    print(f"{name:35s} {row['t_us']:7.2f} μs")
```

Each `Anchors()` field (M, G, alpha, bw, dims, L, alpha_inner, bw_inner, ...) is overridable; pass the modified anchor back into `run_all` to re-evaluate every primitive.

---

## Canonical regression anchors

The notebooks reproduce these exactly — useful as test oracles when changing cost code:

**Ideal (`documentation/modeling/04_in_network_collectives.md` §3.1)** — N=512, M=16 MB, α=0.5 μs, BW=900 GB/s:

| Algorithm | Cost |
|---|---|
| star + ring AR | 546 μs |
| star + DBT (n_β=2 dual-touch) | 45 μs |
| 512-port star + NVLS (INC) | 18.8 μs |
| torus 8³ + dim-decomp ring | 57 μs |

**Realistic η (`documentation/modeling/05_contention_and_congestion.md` §6.1)**, same anchor under the §4.1 calibration:

| Algorithm | Profile | Cost |
|---|---|---|
| star + NVLS (INC) | nvls (η_α=1.00, η_β=0.52) | 35 μs |
| star + DBT (dual-touch) | crossbar (η_α=1.00, η_β=0.80) | 53 μs |
| torus 8³ + dim-decomp | torus (η_α=1.20, η_β=0.60) | 84 μs |

---

## Project conventions

- **Docs are the source of truth.** When a formula in the docs has a "practice caveat," the code still implements the pure α-β formula — caveats belong in docstrings, not numerical results.
- **Core stays stdlib-only.** Notebooks may import pandas / matplotlib.
- **η profiles are per-fabric calibrations**, not free dials. `nvls` only applies to INC AR; `torus` only to torus algorithms; `crossbar` to software AR on NVSwitch. See `documentation/modeling/05_contention_and_congestion.md` §4.1.
- **Pipelined vs non-pipelined** is exposed as a `pipelined: bool = False` flag on BC / Reduce / DBT AR. Default = literal P=1 schedule; `pipelined=True` = the asymptotic P→P\* limit (Appendix C of `documentation/modeling/01_collective_algorithms.md`).

---

## Status

- Cost primitives covering BC, Reduce, AR, AG, RS, A2A across star / torus / mesh / fat-tree / in-network paths.
- Five tutorial notebooks (00–04) reproducing the canonical anchors and sweeping the four major trade-off axes (M, G, η, L).
- Regression-anchored against `documentation/modeling/04_in_network_collectives.md` §3.1 and `documentation/modeling/05_contention_and_congestion.md` §6.1.
