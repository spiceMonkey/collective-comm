# Collectives Explainer Series

**Author:** Yue Lu  
**Date:** April 2026  

A walkthrough of collective communication for distributed GPU workloads — from the topology-free α-β cost model down to how dynamic contention shifts the Pareto rankings on real clusters. The primitives and cost models here apply to LLM training, LLM inference, and HPC workloads alike; where worked examples use inference-scale message sizes or LLM parallelism mappings, they're concrete illustrations, not restrictions on scope. Every file in this folder is self-contained; you can start with `01_collective_algorithms.md`, or jump straight to any topic-specific note once you know the vocabulary. `00_summary.md` is a one-page cheatsheet of symbols, primitives, and the canonical α-β formulas for readers who already know the material and want fast lookup.

## Reading order

```
    01_collective_algorithms.md          ← start here
              │
              ▼
    02_topology_mapping.md               (single-tier: star, torus, mesh)
              │
              ▼
    03_hierarchical_topologies.md        (multi-tier: fat-tree/Clos + composition)
              │
    ┌─────────┴────────────┐
    ▼                      ▼
  04_in_network_         05_contention_
  collectives.md          and_congestion.md
```

**Branch structure.** Read `01_collective_algorithms.md` → `02_topology_mapping.md` → `03_hierarchical_topologies.md` first (topology: single-tier, then multi-tier). After that, two independent branches:

- `04_in_network_collectives.md` deepens SHARP / NVLS / Quantum SHARP — in-network reduction and switch multicast, on both star (`02_topology_mapping.md` §2) and fat-tree spine (`03_hierarchical_topologies.md` §1).
- `05_contention_and_congestion.md` extends the ideal-model ladders from `02_topology_mapping.md` and `03_hierarchical_topologies.md` under realistic contention coefficients $\eta_\alpha, \eta_\beta$; per-tier $\eta$ profile for hierarchical fabrics.

Read `04_in_network_collectives.md` for SHARP's $O(N) \to O(1)$ hop-count collapse; `05_contention_and_congestion.md` for how real-cluster contention changes ideal-model rankings.

## File index

| File | Topic | Prereq |
|---|---|---|
| `00_summary.md` | One-page cheatsheet: symbol table, seven primitives, per-algorithm $(n_\alpha, n_\beta)$ table, per-topology specializations, hierarchical composition rule, INC effects, η-realistic form, $N = 512$ anchor numbers | familiarity with the rest of the series |
| `01_collective_algorithms.md` | α-β cost model; seven primitives (BC, Reduce, AR, RS, AG, A2A, P2P); binomial / pipelined BC / Reduce; ring / tree / recursive-doubling / Rabenseifner AR; mapping to TP/EP/SP/PP; algbw/busbw conventions | None |
| `02_topology_mapping.md` | Three-topology catalog of single-tier scale-up fabrics: star, torus, mesh. Per-topology cost derivations; torus dim-decomp AR with 2×2 worked example; side-by-side comparison at $N = 512$ | `01_collective_algorithms.md` |
| `03_hierarchical_topologies.md` | Multi-tier Clos / fat-tree (§1, including the NVL72 SuperPOD case study); composition rules (RS → sub-AR → AG; A2A outlier) (§2); INC and per-tier $\eta$ in hierarchies (§3); rail-optimized SuperPOD topology and k-ary fat-tree appendices | `02_topology_mapping.md` |
| `04_in_network_collectives.md` | SHARP / NVLS / Quantum SHARP — in-network reduction (switch ALU), switch multicast, and emerging HW A2A as distinct capabilities; how $n_\alpha$ collapses from $O(N)$ to $O(1)$ | `01_collective_algorithms.md`, `02_topology_mapping.md` §2, `03_hierarchical_topologies.md` §1 |
| `05_contention_and_congestion.md` | Contention coefficients $\eta_\alpha, \eta_\beta$; single-tier and per-tier calibration (fat-tree oversubscription $s \to \eta_\beta \leq 1/s$); re-running $N = 512$ under realistic $\eta$ | `02_topology_mapping.md` §5, `03_hierarchical_topologies.md` §1 |

## Primitive → section map

| Primitive | Introduced in |
|---|---|
| Point-to-point (p2p) | `01_collective_algorithms.md` §8 |
| Broadcast (ring-chain / binomial tree / pipelined tree) | `01_collective_algorithms.md` §3 |
| Reduce (binomial / pipelined) | `01_collective_algorithms.md` §4 |
| Ring all-reduce | `01_collective_algorithms.md` §5.1 |
| Double binary tree all-reduce (NCCL) | `01_collective_algorithms.md` §5.2 |
| Simple recursive-doubling AR | `01_collective_algorithms.md` App. B.1 |
| Rabenseifner halving-doubling AR | `01_collective_algorithms.md` App. B.2 |
| Ring all-gather / reduce-scatter | `01_collective_algorithms.md` §6 |
| PAT all-gather / reduce-scatter (NCCL 2.23+, scale-out) | `01_collective_algorithms.md` App. A |
| Recursive-doubling AG / recursive-halving RS | `01_collective_algorithms.md` App. B.4 |
| Ring-relay all-to-all | `01_collective_algorithms.md` §7.1 |
| Pairwise direct-send all-to-all | `01_collective_algorithms.md` §7.2 |
| Bruck all-to-all | `01_collective_algorithms.md` App. B.5 |
| Switch-multicast-assisted AG / BC | `04_in_network_collectives.md` §1.2 |
| Torus BC / Reduce (dim-decomposed) | `02_topology_mapping.md` §3.2, §3.3 |
| Torus dim-decomposed ring AR | `02_topology_mapping.md` §3.4 |
| Torus dim-decomposed Rabenseifner AR | `02_topology_mapping.md` App. A |
| Torus dim-decomposed AG / RS | `02_topology_mapping.md` §3.5 |
| Torus A2A (bisection-limited) | `02_topology_mapping.md` §3.6 |
| Full mesh (direct, no switch) | `02_topology_mapping.md` §4.1 |
| $k$-D mesh (torus without wraparound) | `02_topology_mapping.md` §4.2 |
| Fat-tree / Clos (leaf-spine, 3-tier) | `03_hierarchical_topologies.md` §1 |
| Hierarchical AR (RS → sub-AR → AG) | `03_hierarchical_topologies.md` §2.1 |
| In-network AR (SHARP / NVLS / INC) | `04_in_network_collectives.md` |

## What's *not* here

- **Formal derivations** of the analytical cost model and its integration into a specific performance tool — those live in the tool-specific documentation, not in this general explainer series.
- **Workload-specific end-to-end latency models** (inference decode / prefill, training iteration time) — the collectives here are one ingredient; the full pipeline treatment is out of scope.
- **Runnable benchmarks or Pareto sweeps** — this folder is reading material. Calibration experiments belong wherever the reader's performance tool lives.

## For readers vs for practitioners

- **Readers** who want intuition and visuals: `01_collective_algorithms.md` → `02_topology_mapping.md` → `03_hierarchical_topologies.md` → (`04_in_network_collectives.md` and `05_contention_and_congestion.md` by interest).
- **Practitioners** who want to plug numbers into a cost formula: skim `00_summary.md` for the symbol table and per-algorithm $(n_\alpha, n_\beta)$, then jump to the cost summary in `02_topology_mapping.md` §5.1 (single-tier ideal) and `05_contention_and_congestion.md` §5 (realistic); the formulas are stated in-line and self-contained. For multi-tier fabrics, `03_hierarchical_topologies.md` §2 extends the single-tier formulas with the hierarchical composition rule.
- **Reviewers / decision-makers** comparing architectures: start at `02_topology_mapping.md` §5.1 (ideal single-tier formulas) and `04_in_network_collectives.md` §3.1 (concrete N=512 ladder with INC), then `05_contention_and_congestion.md` §5 (realistic), and cross-check the margin-compression discussion in `05_contention_and_congestion.md` §5.3–§5.4. For composing INC across tiers in a hierarchy, `03_hierarchical_topologies.md` §3 covers SHARP at the inner / outer tier and per-tier $\eta$.
