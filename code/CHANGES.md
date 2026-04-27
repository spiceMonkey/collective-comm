# `collective_cost.py` rewrite — change log

**File:** `code/core/collective_cost.py`
**Spec source of truth:** `documentation/modeling/01_collective_algorithms.md` through `05_contention_and_congestion.md`.

This document describes a backward-incompatible rewrite of `collective_cost.py` to align it with the modeling series and add the missing primitives needed for full-coverage sweep notebooks. If you import from this module elsewhere (e.g., from `llm_perf/` or `calculon/`), use this doc to migrate.

---

## 1. Bug fixes

### 1.1 `tree_all_reduce` — wrong BW coefficient

**Old:**

```python
return 2 * math.ceil(math.log2(G)) * alpha_s + 2 * ((G - 1) / G) * (M / bw_Bps)
```

The BW term `2(G-1)/G · M/BW` matched neither DBT form derived in `01_collective_algorithms.md` §5.2.

**New:** `pipelined: bool = False` flag, two regimes:

- `pipelined=False` (default, literal P=1 schedule):
  `t = 2⌈log₂G⌉·α + ⌈log₂G⌉ · M/BW`
- `pipelined=True` (asymptotic P→P* limit):
  `t ≈ 2⌈log₂G⌉·α + M/BW`

The docstring also notes that `04_in_network_collectives.md` §3.1's `n_β=2` "dual-touch" anchor (45 μs at N=512) uses a third form (`2·log_G·α + 2·M/BW`) that callers can compute directly when reproducing that ladder.

### 1.2 `torus_moe_all_to_all` (renamed `torus_a2a`) — wrong mesh diameter

**Old:**

```python
diam = sum(d // 2 for d in dims)        # used for both torus and mesh
denom = 8.0 if wraparound else 4.0
```

The denominator switch was right; the diameter switch was missing.

**New:**

```python
if wraparound:
    diam = sum(d // 2 for d in dims)    # torus diameter
    denom = 8.0
else:
    diam = sum(d - 1 for d in dims)     # mesh (open-line) diameter
    denom = 4.0
```

Per `02_topology_mapping.md` §3.6 (torus) / §4.2 (mesh).

### 1.3 `pairwise_a2a` — stripped implicit MoE Dispatch+Combine factor of 2

**Old:** baked in a 2× factor on both α and BW, returning the round-trip cost. This made it inconsistent with `inc_a2a`, `torus_a2a`, etc., which all model a single A2A direction. Non-MoE callers were silently overcharged 2×.

**New:** matches the base form from `01_collective_algorithms.md` §7.2:

```
t = (G-1)·α + (G-1)/G · M/BW
```

MoE Dispatch+Combine callers double externally: `2 * pairwise_a2a(...)`.

---

## 2. Convention change (AG / RS)

AG and RS now follow the doc convention, where `M` is the **per-rank full payload** (gathered output for AG; pre-scatter input for RS) — equivalent to `G × per-rank shard`. Previously `M` was the per-rank shard.

Affected functions: `ring_all_gather`, `ring_reduce_scatter`, `torus_all_gather`, `torus_reduce_scatter`, `inc_all_gather`, `inc_reduce_scatter`, plus the new `recursive_doubling_all_gather`, `recursive_halving_reduce_scatter`, `pat_all_gather`, `pat_reduce_scatter`.

**Migration:** if your call site passed `shard_bytes`, change to `G * shard_bytes` (or equivalently the full per-rank pre-scatter / post-gather size). The formulas now read identically to the docs:

- Ring AG / RS: `(G-1)·α + (G-1)/G · M/BW`
- Torus AG / RS: `Σ(d_i-1)·α + (N-1)/N · M/BW`
- INC AG / RS: `2·α_switch + (G-1)/G · M/BW`

AR, BC, Reduce, A2A, P2P conventions unchanged (already used full per-rank payload).

---

## 3. Removed

| Removed symbol | Reason | Migration |
|---|---|---|
| `ring_moe_all_to_all` (alias for `pairwise_a2a`) | The "ring" name was misleading per `01_collective_algorithms.md` §7 (the underlying primitive is pairwise direct-send, not a ring). The implicit 2× factor (see §1.3) is also gone. | Use `pairwise_a2a` directly; multiply by 2 externally for MoE Dispatch+Combine. |
| `tree_moe_all_to_all` (deprecated wrapper, emitted `DeprecationWarning`) | Tree A2A is reference-only and not derived in any modeling doc. | Use `pairwise_a2a` for star A2A; `inc_a2a` for HW A2A on Tomahawk Ultra / Rubin; `bruck_a2a` for the log-round reference. |
| `aggregate_per_stage` | Referenced `decode.md` / `prefill.md` sections that don't exist in this repo's `documentation/modeling/`. The function was an LLM-perf-side glue helper, not a collective primitive. | Re-add downstream (in `llm_perf/`) if needed. The structure was: `(L/PP)·(n_TP·t_TP + n_SP·t_SP) + (L_moe/PP)·(n_EP·t_EP) + t_PP`. |

---

## 4. Renamed

| Old name | New name | Reason |
|---|---|---|
| `torus_moe_all_to_all` | `torus_a2a` | Drops the MoE-specific framing; the function models torus A2A regardless of workload. |

---

## 5. Added primitives

All formulas live in `documentation/modeling/00_summary.md` (cheatsheet) with derivations in 01-05.

### 5.1 Broadcast

| Function | Formula | Source |
|---|---|---|
| `ring_broadcast(M, G, α, BW, pipelined=False)` | non-pipe: `(G-1)α + (G-1)·M/BW`; pipe: `(G-1)α + M/BW` | `01_collective_algorithms.md` §3.1 |
| `tree_broadcast(M, G, α, BW, pipelined=False)` | non-pipe: `⌈log₂G⌉·α + ⌈log₂G⌉·M/BW`; pipe: `⌈log₂G⌉·α + M/BW` | `01_collective_algorithms.md` §3.2 |
| `inc_broadcast(M, α_s, BW)` | `2·α_switch + M/BW` | `01_collective_algorithms.md` §3.3, `04_in_network_collectives.md` §1.2 |
| `torus_broadcast(M, dims, α, BW, wraparound=True, pipelined=False)` | torus: `Σ⌊d_i/2⌋·α + (1 or Σ⌊d_i/2⌋)·M/BW`; mesh: `Σ(d_i-1)·α + (1 or Σ(d_i-1))·M/BW` | `02_topology_mapping.md` §3.2 / §4.2 |

### 5.2 Reduce

| Function | Formula | Source |
|---|---|---|
| `ring_reduce(M, G, α, BW, pipelined=False)` | same as `ring_broadcast` (time-reverse) | `01_collective_algorithms.md` §4.1 |
| `tree_reduce(M, G, α, BW, pipelined=False)` | same as `tree_broadcast` (time-reverse) | `01_collective_algorithms.md` §4.2 |
| `inc_reduce(M, α_s, BW)` | `2·α_switch + M/BW` | `01_collective_algorithms.md` §4.3 |
| `torus_reduce(M, dims, α, BW, wraparound=True, pipelined=False)` | same shape as `torus_broadcast` | `02_topology_mapping.md` §3.3 / §4.2 |

### 5.3 All-reduce

| Function | Formula | Source |
|---|---|---|
| `rabenseifner_all_reduce(M, G, α, BW)` | `2⌈log₂G⌉·α + 2(G-1)/G · M/BW` | `01_collective_algorithms.md` App. B.2 (power-of-2 G; reference, not shipped by NCCL) |

### 5.4 All-gather / Reduce-scatter

| Function | Formula | Source |
|---|---|---|
| `recursive_doubling_all_gather(M, G, α, BW)` | `⌈log₂G⌉·α + (G-1)/G · M/BW` | `01_collective_algorithms.md` App. B.4 |
| `recursive_halving_reduce_scatter(M, G, α, BW)` | `⌈log₂G⌉·α + (G-1)/G · M/BW` | `01_collective_algorithms.md` App. B.4 |
| `pat_all_gather(M, G, α, BW)` | `⌈log₂G⌉·α + (G-1)/G · M/BW` | `01_collective_algorithms.md` App. A (NCCL 2.23+ scale-out, 1 rank/node) |
| `pat_reduce_scatter(M, G, α, BW)` | `⌈log₂G⌉·α + (G-1)/G · M/BW` | `01_collective_algorithms.md` App. A |

### 5.5 All-to-all

| Function | Formula | Source |
|---|---|---|
| `ring_relay_a2a(M, G, α, BW)` | `(G-1)·α + (G-1)/G · M/BW` | `01_collective_algorithms.md` §7.1 (α-β-equivalent to pairwise; ships on bisection-limited fabrics) |
| `bruck_a2a(M, G, α, BW)` | `⌈log₂G⌉·α + ⌈log₂G⌉/2 · M/BW` | `01_collective_algorithms.md` App. B.5 (reference; not shipped by NCCL/RCCL) |

### 5.6 Hierarchical composition

| Function | Purpose | Source |
|---|---|---|
| `hierarchical_all_reduce(t_rs_inner, t_ar_outer, t_ag_inner)` | Sum of three pre-computed phases (RS → outer AR → AG); caller picks per-tier algorithm and `(α, BW)`. | `03_hierarchical_topologies.md` §2.1 |
| `hierarchical_all_reduce_ring_ring(M, N, L, α_inner, BW_inner, α_outer, BW_outer)` | Convenience: ring-on-ring composition with payload telescoping (`M·L/N` at outer tier). Reduces to flat ring AR for `L=1` or `L=N`. | `03_hierarchical_topologies.md` §2.1 (NVL72+IB SuperPOD case study) |
| `hierarchical_all_gather(t_ag_inner, t_ag_outer)` | AG ≡ inner AG → outer AG (no telescoping). | `03_hierarchical_topologies.md` §2.1 summary table |
| `hierarchical_reduce_scatter(t_rs_outer, t_rs_inner)` | RS ≡ outer RS → inner RS (time-reverse of hierarchical AG). | `03_hierarchical_topologies.md` §2.1 |

### 5.7 Realistic-cost helpers

| Function | Purpose | Source |
|---|---|---|
| `apply_eta(α, BW, η_α=1.0, η_β=1.0) → (α_eff, BW_eff)` | Pre-discount `(α, BW)` by contention coefficients before passing into any primitive. Calibration profile in the docstring. | `05_contention_and_congestion.md` §4 |
| `realistic_cost(cost_fn, *args, η_α=1.0, η_β=1.0)` | Wrapper that calls `apply_eta` on the last two positional args (the standard `(α_s, bw_Bps)` slot in every primitive in this module) and dispatches. Convenience for sweep notebooks. | `05_contention_and_congestion.md` §4 |

---

## 6. Stale doc reference cleanup

Every formula source comment now points to the current modeling docs. Old → new mapping:

| Old reference (in docstrings) | New reference |
|---|---|
| `decode.md §5.1` (P2P) | `01_collective_algorithms.md` §8 |
| `decode.md §5.3.1` (ring AR) | `01_collective_algorithms.md` §5.1 |
| `collectives.md §4.1` (DBT AR / ring AG-RS) | `01_collective_algorithms.md` §5.2 / §6 |
| `collectives.md §5.1` (pairwise A2A) | `01_collective_algorithms.md` §7.2 |
| `collectives/03_in_network_collectives.md §1.2-1.3` | `04_in_network_collectives.md` §1.1, §1.2, §1.4 |
| `collectives.md §4.4` (INC AG/RS) | `04_in_network_collectives.md` §1.4 |
| `collectives.md §5.2` (torus A2A) | `02_topology_mapping.md` §3.6 |
| `collectives.md §5.4` (HW A2A) | `04_in_network_collectives.md` §1.3 |
| `collectives.md §3.2` (torus AR) | `02_topology_mapping.md` §3.4 |
| `decode.md §5.5` (per-stage aggregate) | removed — function deleted |

Per the repo-level memory rule, references use full filenames (`01_collective_algorithms.md §5.1`), never shorthand (`01 §5.1`).

---

## 7. Validation

`python code/core/collective_cost.py` runs a `_self_test()` that reproduces the canonical regression anchors from `CLAUDE.md`:

**Ideal** (N=512, M=16 MB, α=0.5 μs, BW=900 GB/s):

| Anchor | Target | Reproduced |
|---|---|---|
| `04_in_network_collectives.md` §3.1 star + ring AR | 546 μs | 546.49 μs |
| `04_in_network_collectives.md` §3.1 star + DBT (dual-touch n_β=2) | 45 μs | 44.56 μs |
| `04_in_network_collectives.md` §3.1 hyp. 512-port star + NVLS | 18.8 μs | 18.78 μs |
| `04_in_network_collectives.md` §3.1 torus 8³ dim-decomp ring AR | 57 μs | 56.49 μs |

**Realistic η** (`05_contention_and_congestion.md` §5.1):

| Anchor | Target | Reproduced |
|---|---|---|
| INC (η_β = 0.52) | 35 μs | 35.19 μs |
| DBT (η_β = 0.80, dual-touch n_β=2) | 53 μs | 53.44 μs |
| Torus (η_α = 1.20, η_β = 0.60) | 84 μs | 84.34 μs |

Cross-primitive sanity (`04_in_network_collectives.md` §3.3 anchors):

| Anchor | Target | Reproduced |
|---|---|---|
| Star ring AG | 273 μs | 273.24 μs |
| Star rec-doub AG | 22.2 μs | 22.24 μs |
| Star INC AG | 18.7 μs | 18.74 μs |
| Torus 8³ dim-decomp AG | 28.2 μs | 28.24 μs |
| Star pipelined-tree BC | 22.3 μs | 22.28 μs |
| Star INC BC | 18.3 μs | 18.78 μs |
| Torus 8³ dim-decomp BC (pipelined) | 23.8 μs | 23.78 μs |
| Star pairwise A2A | 273 μs | 273.24 μs |
| Star HW A2A | ~19 μs | 18.24 μs |
| Torus 8³ A2A | 23.8 μs | 23.78 μs |
| Mesh 8³ A2A (= 2× torus) | 47.6 μs | 46.06 μs |

Hierarchical NVL72+IB AR (`03_hierarchical_topologies.md` §2.1, N=144, L=2): target ~114 μs, reproduced 114.51 μs.

---

## 8. API summary table

| Group | Functions |
|---|---|
| **P2P** | `p2p_hop` |
| **BC** | `ring_broadcast`, `tree_broadcast`, `inc_broadcast`, `torus_broadcast` |
| **Reduce** | `ring_reduce`, `tree_reduce`, `inc_reduce`, `torus_reduce` |
| **AR** | `ring_all_reduce`, `tree_all_reduce`, `rabenseifner_all_reduce`, `inc_all_reduce`, `torus_all_reduce` |
| **AG** | `ring_all_gather`, `recursive_doubling_all_gather`, `pat_all_gather`, `inc_all_gather`, `torus_all_gather` |
| **RS** | `ring_reduce_scatter`, `recursive_halving_reduce_scatter`, `pat_reduce_scatter`, `inc_reduce_scatter`, `torus_reduce_scatter` |
| **A2A** | `pairwise_a2a`, `ring_relay_a2a`, `bruck_a2a`, `inc_a2a`, `torus_a2a` |
| **Hierarchical** | `hierarchical_all_reduce`, `hierarchical_all_reduce_ring_ring`, `hierarchical_all_gather`, `hierarchical_reduce_scatter` |
| **Realistic η** | `apply_eta`, `realistic_cost` |
