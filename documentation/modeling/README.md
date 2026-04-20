# Collectives Explainer Series

**Author:** Yue Lu  
**Date:** April 2026  

A walkthrough of collective communication for distributed GPU workloads — from the topology-free α-β cost model down to how dynamic contention shifts the Pareto rankings on real clusters. The primitives and cost models here apply to LLM training, LLM inference, and HPC workloads alike; where worked examples use inference-scale message sizes or LLM parallelism mappings, they're concrete illustrations, not restrictions on scope. Every file in this folder is self-contained; you can start with 01, or jump straight to any topic-specific note once you know the vocabulary.

## Reading order

```
    01_collective_algorithms.md          ← start here
              │
              ▼
    02_topology_mapping.md
              │
    ┌─────────┴─────────────┐
    ▼                       ▼
  03_in_network_          04_contention_
  collectives.md          and_congestion.md
```

**Branch structure.** Read 01 → 02 first. After that, the two branches are independent:

- **03** deepens §2 of 02 (SHARP / NVLS on star topologies; in-network reduction mechanics).
- **04** extends §4 of 02 (re-runs the $N = 512$ comparison under realistic contention coefficients).

You only need to read 03 if you want to understand SHARP's $O(N) \to O(1)$ hop-count collapse, and 04 if you want to know how much the ideal-model rankings change under real-cluster contention.

## File index

| File | Topic | Prereq |
|---|---|---|
| `01_collective_algorithms.md` | α-β cost model; the seven primitives; ring / tree AR worked examples; mapping to TP/EP/SP/PP | None |
| `02_topology_mapping.md` | Star / torus primer; per-topology cost derivations; torus dim-decomp AR mechanism with 2×2 worked example; side-by-side comparison at $N = 512$ | 01 |
| `03_in_network_collectives.md` | SHARP / NVLS / Quantum SHARP — how switch-resident reduction collapses $n_\alpha$ from $O(N)$ to $O(1)$ | 01, 02 §2 |
| `04_contention_and_congestion.md` | Contention coefficients $\eta_\alpha, \eta_\beta$; calibrating from public benchmarks; re-running $N = 512$ under realistic $\eta$ | 02 §4 |

## Primitive → section map

| Primitive | Introduced in |
|---|---|
| Point-to-point (p2p) | 01 §6 |
| Ring all-reduce | 01 §3.1 |
| Double binary tree all-reduce (NCCL) | 01 §3.2 |
| Simple recursive-doubling AR | 01 App. A.1 |
| Rabenseifner halving-doubling AR | 01 App. A.2 |
| Ring all-gather / reduce-scatter | 01 §4.1 |
| PAT all-gather / reduce-scatter (NCCL 2.23+, scale-out) | 01 App. B.2 |
| Recursive-doubling AG / recursive-halving RS | 01 App. B.1 |
| Pairwise direct-send all-to-all | 01 §5.1 |
| Bruck all-to-all | 01 App. B.3 |
| Torus dim-decomposed ring AR | 02 §3.1 |
| Torus dim-decomposed Rabenseifner AR | 02 App. A |
| Torus dim-decomposed AG / RS | 02 §3.2 |
| Torus A2A (bisection-limited) | 02 §3.3 |
| In-network AR (SHARP / NVLS / INC) | 03 |

## What's *not* here

- **Formal derivations** of the analytical cost model and its integration into a specific performance tool — those live in the tool-specific documentation, not in this general explainer series.
- **Workload-specific end-to-end latency models** (inference decode / prefill, training iteration time) — the collectives here are one ingredient; the full pipeline treatment is out of scope.
- **Runnable benchmarks or Pareto sweeps** — this folder is reading material. Calibration experiments belong wherever the reader's performance tool lives.

## For readers vs for practitioners

- **Readers** who want intuition and visuals: 01 → 02 → (any of 03, 04 by interest).
- **Practitioners** who want to plug numbers into a cost formula: jump to the cost tables in 02 §4 (ideal) and 04 §4 (realistic); the formulas are stated in-line and self-contained.
- **Reviewers / decision-makers** comparing architectures: start at 02 §4 (ideal comparison), then 04 §4 (realistic), and cross-check the margin-compression discussion in 04 §4.3–§4.4.
