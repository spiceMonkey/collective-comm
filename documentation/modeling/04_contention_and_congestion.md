# Contention and Congestion: From Algorithmic Ceilings to Realized Performance

**Author:** Yue Lu  
**Date:** April 2026  

`02_topology_mapping.md` scored each collective as if it ran alone on a perfectly-scheduled fabric; `03_in_network_collectives.md` tightened that upper bound to the algorithmic ceiling ($n_\alpha, \mathrm{BW_{eff}}$) under in-network reduction. Real deployments fall below both — some of the gap is baked into the silicon and fabric (link utilization ceilings, switch queueing, cross-tier backpressure), and some is left on the table by the runtime scheduler (which rank layout the allocator produces, which concurrent groups share physical links, how expert traffic is routed). This note splits the gap along that axis, introduces per-fabric contention coefficients $\eta_\alpha \ge 1$ and $\eta_\beta \in (0, 1]$ as the scalar aggregate, and re-runs the $N = 512$ comparison under realistic $\eta$ to show how much the ideal-model margins compress.

# Table of Contents

1. [From ceilings to realized](#1-from-ceilings-to-realized)
2. [Hardware-level inefficiencies](#2-hardware-level-inefficiencies)
3. [Software/scheduling-level inefficiencies](#3-softwarescheduling-level-inefficiencies)
4. [Contention coefficients $\eta_\alpha$ and $\eta_\beta$](#4-contention-coefficients-η_α-and-η_β)
5. [Calibrating $\eta$ from public benchmarks](#5-calibrating-η-from-public-benchmarks)
6. [Re-running $N = 512$ under realistic $\eta$](#6-re-running-n--512-under-realistic-η)
7. [What the coefficient model doesn't capture](#7-what-the-coefficient-model-doesnt-capture)
8. [When $\eta$ is load-bearing](#8-when-η-is-load-bearing)
9. [Further reading](#further-reading)

---

## 1. From ceilings to realized

The cost formulas in `02_topology_mapping.md` and the INC ceilings in `03_in_network_collectives.md §1.3` both describe upper bounds. Realized deployment performance sits below those ceilings, and the gap decomposes cleanly into two sources:

- **Hardware-level inefficiencies** — what an isolated, perfectly-scheduled collective still pays because the fabric cannot sustain its nominal link BW or switch latency under realistic load (§2).
- **Software/scheduling-level inefficiencies** — what the runtime scheduler leaves on the table by accepting an off-prefix rank layout, scheduling concurrent groups that share physical links, or routing expert traffic non-uniformly (§3).

A useful way to keep the two separated: if the inefficiency persists when a single, isolated collective runs on a freshly-booted cluster, it's hardware-level. If it disappears when the scheduler makes better choices without touching the silicon, it's software/scheduling-level.

**Scope assumption.** We assume the deployment has already selected the right algorithm for its fabric at design time (DBT or NVLS on star, dim-decomp on torus, INC where available). Algorithm pairing is a one-time design decision, not a runtime contention effect; a well-designed deployment pays it only in the form of "pick the right one," after which the $(\eta_\alpha, \eta_\beta)$ model below prices what's left.

Both sources show up in the cost model as a combination of "more $\alpha$" and "less effective BW." §4 introduces the scalar $(\eta_\alpha, \eta_\beta)$ coefficients that aggregate them per (fabric, collective).

---

## 2. Hardware-level inefficiencies

These are unavoidable without different silicon or different fabric architecture. Each shows up in the coefficient model as a floor on $\eta_\beta$ or a floor on $\eta_\alpha$ that the software stack cannot drive away.

### 2.1 Link BW utilization ceiling

Nominal per-link bandwidth is set by pin count $\times$ signaling rate, but the fraction a collective actually sees is discounted by framing overheads, endpoint DMA efficiency, and flow-control headroom required to keep the link from stalling. NCCL-tests on an H100 NVLink4 node measures AR busbw $\approx 360\,\mathrm{GB/s}$ against $450\,\mathrm{GB/s}$ raw unidirectional link BW — a baseline $\eta_\beta \approx 0.80$ that shows up *without any concurrency at all*. This is the floor the software stack cannot beat: even a single TP AR on an otherwise-idle node pays it.

### 2.2 Switch/fabric congestion under load

Wormhole / cut-through routing keeps per-message BW close to peak once a path is established. Queue delay, not per-hop latency, is what inflates under load: as fabric utilization crosses $\sim 80\%$, head-of-line blocking and arbitration delay at each switch grow super-linearly. Effective $\alpha$ per message can double or triple when the fabric carries multiple concurrent traffic classes — training mixes gradient AR with activation AG/RS and DP AR; inference mixes TP AR with KV streaming and scheduler messages; HPC mixes MPI collectives with RDMA p2p. The α-β model doesn't price this implicitly, so congestion shows up as $\eta_\alpha > 1$.

**Hierarchical specialization.** Multi-tier Clos and fat-tree fabrics typically oversubscribe upper tiers relative to leaves (e.g., 2:1 or 4:1 at the aggregation layer). When an upper-tier link saturates, credit-based flow control pushes backpressure into lower tiers — the same congestion mechanism, but per-tier $\eta$ treats tiers as independent and under-counts the coupling. Tight flow-controlled fabrics (InfiniBand, RoCE with PFC) violate the independence assumption; loose store-and-forward or timeout-based fabrics approximate it better. For scale-out INC paths (`03 §2.2`), cross-tier coupling is what makes the $2k\alpha_\mathrm{switch}$ floor grow faster than the per-tier walk under saturation.

---

## 3. Software/scheduling-level inefficiencies

These are fixable in the scheduler — different rank layout, different group placement, different routing policy — without touching the hardware. They show up as an *additional* discount on top of the hardware floor: if the scheduler picks well, realized $\eta$ approaches the hardware-only floor; if it picks badly, realized $\eta$ can be much worse.

### 3.1 Off-prefix group layouts

Torus dim-decomp AR hits $n_\alpha = 2 \sum (D_i - 1)$ only when the collective group's ranks are contiguous along dim prefixes. A group of 16 ranks on an $8 \times 8 \times 8$ torus where the ranks are $\{(0,0,0), (1,0,0), \ldots, (7,0,0), (0,1,0), \ldots\}$ is prefix-contiguous along dim-0 — the algorithm runs as a single 8-ring.

But SLURM-style scatter allocation, or a job that inherits a random-looking rank layout, produces groups where ranks are spread across dim coordinates. Each algorithmic hop becomes a multi-hop path in the physical topology. The formula degrades back to the flat-ring bound ($n_\alpha \approx 2(N-1)$), often 2-4× the ideal. This is a **scheduler choice**, not a hardware limit: prefix-aligned allocation policies exist, and when available they eliminate this penalty entirely.

**Residual unavoidable case.** Two cases escape even an optimal allocator: (a) **group-shape mismatch** — a TP = 6 group on an $8 \times 8 \times 8$ torus has no prefix-contiguous placement because 6 doesn't divide any dim; (b) **multi-tenant fragmentation** — a shared cluster with asynchronous job arrivals eventually runs out of prefix-aligned free slots and must place the next job off-prefix. Both are set by workload shape × topology geometry, not by the scheduler.

### 3.2 Concurrent collective groups

If the cluster runs DP = 8 replicas each doing TP = 8 AR simultaneously, where those 8 groups land on the physical fabric matters. On a **star** with $N = 64$ ports, the 8 groups partition into 8 disjoint port octets — each sees full BW with zero interference. **Torus is different**: if the 8 replicas share a common dim (e.g., all TP groups run along the X-axis), those 8 rings physically traverse the same links, and their aggregate BW is split. The ideal dim-decomp AR formula assumes rings occupy disjoint links across dims; concurrent-group layouts can violate that assumption.

Like off-prefix layouts, this is a **layout choice** made by the job scheduler. A layout-aware scheduler that distributes concurrent TP groups across orthogonal dims avoids the penalty; a naive scheduler pays it.

**Residual unavoidable case.** Sharing is forced when: (a) **concurrency exceeds dim count** — 8 concurrent TP groups on a 3D torus have only 3 orthogonal dims to spread across, so at least 3 groups must share a dim; (b) **TP size pins the dim** — TP = 8 on an $8 \times 8 \times 8$ torus must run along a full-dim axis, and every concurrent DP replica doing TP = 8 is forced onto that axis. Once forced, the per-dim BW splits across groups.

### 3.3 Skewed MoE routing

The uniform-bisection A2A bound assumes all $(i, j)$ pairs carry equal traffic. Real MoE traces show 3-10× skew between hot and cold experts — a handful of experts receive the bulk of routed tokens. The bisection cut still carries $N M / 2$ bytes *on average*, but tail latency is set by the hottest link, not the average. Torus has hotspots under skewed routing: hot-expert destinations concentrated along one dim saturate that dim's bisection well before the uniform bound predicts. Star absorbs skew uniformly because every port has the same BW.

Skew is a policy-level artifact of the routing function (top-$k$ gating, capacity factor, expert placement). Load-balancing losses, expert shuffling, and capacity-factor tuning all discount the skew penalty without hardware changes.

**Residual unavoidable case.** Routing is **input-dependent**: even with perfectly load-balanced long-run averages, a single batch can be skewed by what tokens it happens to contain. Average-case throughput can be driven close to uniform, but per-step tail latency always sees some residual hotspot — which is what latency-SLA workloads actually pay on.

---

## 4. Contention coefficients $\eta_\alpha$ and $\eta_\beta$

The idealized collective cost is $t = n_\alpha \cdot \alpha + n_\beta \cdot (M / \mathrm{BW})$. Under the combined discount from §2 and §3, replace with

$$t_{\mathrm{effective}} = n_\alpha \cdot \alpha \cdot \eta_\alpha + n_\beta \cdot \frac{M}{\mathrm{BW} \cdot \eta_\beta}$$

Interpretation:

- $\eta_\alpha \ge 1$ **inflates** the latency term. Aggregates arbitration delay, head-of-line blocking, and cross-tier backpressure under congestion (§2.2), plus any layout-driven extra hops (§3.1).
- $\eta_\beta \in (0, 1]$ **deflates** the effective bandwidth. Aggregates the hardware BW ceiling (§2.1), link sharing between concurrent groups (§3.2), and bisection saturation from skewed traffic (§3.3).

Both default to 1.0, which recovers the ideal model. The cleanest way to apply them is **at the fabric-tier level** — each switch tier or link class carries its own $(\eta_\alpha, \eta_\beta)$, and the primitive cost function sees an "effective" tier with already-discounted $\alpha$ and $\mathrm{BW}$ values. No algorithm-level plumbing is required.

### 4.1 Why a single scalar per (fabric, collective)

The coefficient model is deliberately coarse. Contention in real fabrics depends on layout, payload size, concurrent-job pattern, and dynamic routing state — none of which are scalar. But a two-parameter $(\eta_\alpha, \eta_\beta)$ captures the first-order effect: "how much worse is my collective than the isolated-primitive upper bound?" That's the question a Pareto sweep needs to answer correctly, and scalar coefficients are enough.

Higher-fidelity scoring (packet-level simulation, event-driven fabric models) is the right answer when the coefficient model becomes load-bearing for a production decision. §8 below explains when that's worth doing.

---

## 5. Calibrating $\eta$ from public benchmarks

A representative $\eta$ profile calibrated from public measurements:

| Fabric | $\eta_\alpha$ | $\eta_\beta$ | Source |
|---|---|---|---|
| Crossbar (NVLink+NVSwitch, no SHARP) | 1.00 | 0.80 | NCCL-tests H100 AR busbw 360 GB/s measured vs 450 GB/s peak → 0.80 |
| NVLS (NVLink SHARP, in-network reduction) | 1.00 | 0.52 | NVLS busbw 470 GB/s measured vs DBT 360 GB/s ($\sim 1.3\times$ lift) on same hardware; back-solves to $\eta_\beta^{\mathrm{INC}} \approx 1.3 \cdot \eta_\beta^{\mathrm{DBT}} / 2 = 0.52$ under the $n_\beta^{\mathrm{INC}} = 1$ framing |
| Torus (off-prefix + concurrent groups) | 1.20 | 0.60 | TPU v4 twisted-vs-untwisted 1.63× A2A reported gap, interpreted as upper-bound $\eta_\beta \approx 1/1.63 \approx 0.6$ under adversarial layouts |

Why $\eta_\beta^{\mathrm{INC}} < \eta_\beta^{\mathrm{DBT}}$ even though INC wins: in the coefficient framework, $\eta_\beta$ is measured against *raw* link BW, and the algorithmic factor sits in $n_\beta$. DBT's $n_\beta = 2$ already bakes in a 2× slack, so its $\eta_\beta = 0.80$ applies to a target that was never the full link. INC's $n_\beta = 1$ attempts the full link, and switch-ALU throughput plus multicast contention keep it from fully realizing that — so its $\eta_\beta$ looks lower, even though its *realized* per-rank BW ($0.52 \cdot \mathrm{BW}$) still beats DBT's ($0.80 \cdot \mathrm{BW} / 2 = 0.40 \cdot \mathrm{BW}$) by $\sim 1.3\times$.

One important caveat on these defaults: **they are worst-case-realistic, not typical.** A well-tuned deployment with prefix-aligned groups (§3.1) and SHARP-enabled AR will see much smaller $\eta$ penalties. The profile is a lower bound on performance, not an expectation.

For proprietary or newer fabrics, the right approach is to run NCCL-tests (or an equivalent collective benchmark) under the intended concurrent-group pattern and back out $\eta$ from the ratio of measured `busbw` to the theoretical peak of `algbw`.

---

## 6. Re-running $N = 512$ under realistic $\eta$

Using the realistic profile from §5, re-run the AR ladder from `02 §4` and `03 §3.1` at $N = 512$, $M = 16\,\mathrm{MB}$, $\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$. We score the optimized pairings on each fabric — DBT on star, dim-decomp on torus, NVLS-style INC on the hypothetical single-switch star from `03 §3` — per the scope assumption in §1.

### 6.1 Ideal vs realistic AR cost

| Topology + algo | $n_\alpha$ | $n_\beta$ | Ideal $\alpha$ | Ideal BW | **Ideal total** | $\eta_\alpha$ | $\eta_\beta$ | Realistic $\alpha$ | Realistic BW | **Realistic total** |
|---|---|---|---|---|---|---|---|---|---|---|
| Hypothetical star + NVLS (INC) | 2 | 1 | 1 μs | 17.8 μs | **18.8 μs** | 1.00 | 0.52 | 1 μs | 34.2 μs | **35 μs** |
| Star + DBT (NCCL) | 18 | 2 | 9 μs | 35.5 μs | **45 μs** | 1.00 | 0.80 | 9 μs | 44.4 μs | **53 μs** |
| Torus $8^3$ + dim-ring | 42 | 2 | 21 μs | 35.5 μs | **57 μs** | 1.20 | 0.60 | 25.2 μs | 59.2 μs | **84 μs** |

### 6.2 What changed

- **Star + INC loses $\sim 16\,\mu$s** (86% increase — the largest *relative* penalty of the three). Its ideal BW term is already at the raw-link ceiling ($M / \mathrm{BW}$), so the $\eta_\beta = 0.52$ realization loss nearly doubles the wall-clock cost. Despite this, INC is still the fastest at 35 μs — ~1.5× ahead of star+DBT and ~2.4× ahead of torus.
- **Star + DBT loses $\sim 8\,\mu$s** (18% increase). The BW-dominated cost is inflated by $\eta_\beta = 0.80$; $\alpha$ is unaffected. A flat penalty on an already small BW term.
- **Torus loses $\sim 27\,\mu$s** (47% increase). Both $\eta_\alpha$ and $\eta_\beta$ hit it. The BW term goes from matching star (35.5 μs) to 65% worse (59.2 μs). The $\alpha$ term inflates 20%.

### 6.3 Margin compression

| | Ideal | Realistic | Gap to INC (ideal → realistic) |
|---|---|---|---|
| Star + INC | 18.8 μs | 35 μs | — |
| Star + DBT | 45 μs | 53 μs | $2.4\times$ → $1.5\times$ |
| Torus dim-decomp | 57 μs | 84 μs | $3.0\times$ → $2.4\times$ |

**Takeaway:** the ideal-model ranking is preserved — INC < DBT < torus — but the margins compress *toward* INC as realistic contention kicks in. This is the mirror image of the naive expectation. INC's ideal ceiling is the highest above its realistic cost (86% inflation) because its BW term is closest to the raw-link limit and has nowhere left to hide; DBT and torus start further from their limits and absorb contention into already-padded BW terms. The gap between INC and the next-best software collective compresses from $2.4\times$ to $1.5\times$, and the gap to torus from $3.0\times$ to $2.4\times$ — INC is still the winner, but the ceiling-vs-realized story flagged in `03 §4.2` (limit 6) is where most of the "marketed 2× vs measured 1.3×" NVLS lift goes.

The non-INC margins also compress: star+DBT vs torus goes from 45 vs 57 μs (~25% gap) ideal to 53 vs 84 μs (~60% gap) realistic, because torus pays $\eta$ at both $\alpha$ and BW while star+DBT only pays it on BW. The "star vs torus" decision under realistic $\eta$ is more contention-sensitive than the ideal numbers suggest.

### 6.4 AG / RS and A2A under realistic $\eta$

§6.1–§6.3 covered AR. Re-running the AG / RS and A2A ladders from `02 §4.2–§4.3` and `03 §3.4` at the same anchor ($N = 512$, $M = 16\,\mathrm{MB}$, $\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$) uses a different $\eta_\beta$ for the INC row than AR did: there is no switch-ALU on the AG / RS critical path (only multicast), so INC AG / RS inherits the crossbar-multicast baseline $\eta_\beta = 0.80$ rather than NVLS's 0.52. This is the coefficient-level reflection of the "no BW ceiling lift" structural point from `03 §1.3`: INC AG / RS runs at the same realized BW as a well-scheduled software AG / RS on a full-duplex crossbar.

**AG / RS.**

| Topology + algo | Ideal $\alpha$ | Ideal BW | **Ideal total** | $\eta_\alpha$ | $\eta_\beta$ | Realistic $\alpha$ | Realistic BW | **Realistic total** |
|---|---|---|---|---|---|---|---|---|
| Hypothetical star + INC | 1.0 μs | 17.7 μs | **18.7 μs** | 1.00 | 0.80 | 1.0 μs | 22.2 μs | **23.2 μs** |
| Star + RHD | 4.5 μs | 17.7 μs | **22.2 μs** | 1.00 | 0.80 | 4.5 μs | 22.2 μs | **26.7 μs** |
| Torus $8^3$ + dim-decomp | 10.5 μs | 17.7 μs | **28.2 μs** | 1.20 | 0.60 | 12.6 μs | 29.6 μs | **42.2 μs** |

**A2A.**

| Topology + algo | Ideal $\alpha$ | Ideal BW | **Ideal total** | $\eta_\alpha$ | $\eta_\beta$ | Realistic $\alpha$ | Realistic BW | **Realistic total** |
|---|---|---|---|---|---|---|---|---|
| Star + pairwise (NCCL) | 255.5 μs | 17.7 μs | **273 μs** | 1.00 | 0.80 | 255.5 μs | 22.2 μs | **278 μs** |
| Hypothetical star + INC | — | — | — | — | — | — | — | **N/A on shipping hardware** |
| Torus $8^3$ + bisection (TPU / Trainium) | 6.0 μs | 35.5 μs | **41.5 μs** | 1.20 | 0.60 | 7.2 μs | 59.2 μs | **66.4 μs** |

Three observations:

- **AG / RS margins are already tight and stay tight.** Star INC (23.2 μs) vs star RHD (26.7 μs) vs torus (42.2 μs). The ideal $1.2\times$ INC-vs-RHD gap compresses slightly to $1.15\times$ under realistic $\eta$ — $\alpha$ savings are preserved (INC still wins on $n_\alpha$), but both rows hit the same BW floor so contention doesn't open the gap. AG / RS is a fundamentally $\alpha$-dominated primitive at this $M$; the whole INC-vs-software fight happens on the $\alpha$ side.
- **A2A ranking holds torus under any realistic scoring.** Ideal: torus 41.5 μs < pairwise 273 μs. Realistic: torus 66.4 μs < pairwise 278 μs — ordering preserved, and torus's $\sim 4.2\times$ advantage over the shipped NCCL pairwise widens slightly under $\eta$ because pairwise's $\alpha$-dominated cost absorbs contention only on the tiny BW term while torus pays $\eta$ on both. With no INC path available on shipping hardware, torus is the only primitive that scales A2A without a bisection penalty at large $N$; this is the forward-looking reason Rubin-generation HW-A2A and Tomahawk Ultra's INC A2A exist, and the main A2A mitigation until they ship at scale.
- **Realistic-$\eta$ collapses the cross-primitive ordering into a simple rule.** AR wins by ~$1.5\times$ on INC-capable star; AG / RS wins by ~$1.2\times$ on INC; A2A wins by ~$4.2\times$ on cubic torus (no INC help on shipping HW). The realistic $\eta$ profile doesn't reshuffle the winner within each primitive but does tighten every margin — which directly feeds the topology-choice Pareto sweep of `02 §5` and §8 below.

---

## 7. What the coefficient model doesn't capture

Scalar $(\eta_\alpha, \eta_\beta)$ per (fabric, collective) is a back-of-envelope model. It cannot express:

### 7.1 Layout-dependent contention

Two TP groups whose ranks are all in dim-0 conflict. Two TP groups whose ranks are orthogonal (one in dim-0, one in dim-2) don't. A single $\eta_\beta$ averages over layout. A layout-aware model would need to take the rank-to-coordinate map as input and compute link-sharing per pair of groups.

### 7.2 Payload-size-dependent contention

Contention often manifests at small messages (queue delay dominates) and vanishes at large messages (BW utilization is steady). The coefficient is a workload-average — it over-predicts penalty for large-$M$ collectives (prefill, bulk gradient AR) and under-predicts for small-$M$ collectives (decode AR, latency-critical HPC reductions).

### 7.3 Dynamic patterns

Expert skew, traffic bursts, and temporal collisions between collective steps are all dynamic. A coefficient matched to average load under-reports tail latency, which is often what the user cares about (latency SLA > average throughput).

### 7.4 Cross-tier propagation in multi-tier fabrics

In hierarchical fabrics (multi-tier Clos, fat-tree, or any switched fabric where upper tiers are oversubscribed relative to lower tiers), saturation at an upper tier can push backpressure into lower tiers via credit-based flow control — but a per-tier $\eta$ treats tiers as independent. Tight flow-controlled fabrics violate this assumption; loose store-and-forward or timeout-based fabrics approximate it better.

For any of these, the right answer is a higher-fidelity simulator (packet-level or event-driven, e.g. SST, Booksim). The coefficient model is the layer between "zero contention" and "simulate it" — cheap enough to use in a topology-choice Pareto sweep, principled enough that the rankings mean something.

---

## 8. When $\eta$ is load-bearing

The $(\eta_\alpha, \eta_\beta)$ defaults in §5 are calibrated from public measurements but still broadly applicable. They're good enough for three classes of decision:

- **First-pass topology selection.** "Star vs torus for a new 512-GPU cluster" — the ideal-$\eta$ ranking often holds, but realistic-$\eta$ shifts the margins. Both sweeps should agree on the top choice; if they don't, the decision is contention-sensitive and warrants simulation.
- **Scheduler-policy review.** "Are our rank-allocation and concurrent-group policies prefix-aligned on torus?" — off-prefix (§3.1) and concurrent-group (§3.2) penalties are the primary reason realistic torus $\eta_\beta$ sits at 0.60 vs the 0.80 a star pays; a policy change can recover much of that.
- **Sensitivity analysis.** "How much does our end-to-end step time depend on contention?" (TPOT for inference, step-time for training) — sweep $\eta$ from (1.0, 1.0) ideal to (1.5, 0.4) aggressive and report the envelope.

Cases where $\eta$-based ranking is **not enough** and you should escalate to real simulation or measurement:

- **Deployment budgeting for a new fabric architecture.** "What's the actual max-throughput step time for configuration X?" (TPOT for inference, iteration time for training) — coefficient model will be off by factor-of-2 under unmodeled cross-tier dynamics; need hardware-trace calibration.
- **Production SLA guarantees.** Tail latency dominates; coefficients give average-case estimates.
- **Comparing fabrics within 20% of each other.** The error bars on $\eta$ defaults are wider than 20%; any "ideal-$\eta$ ranking changes under realistic-$\eta$" result should prompt a closer look, not a definitive call.

---

## Further reading

- **`02_topology_mapping.md` §4** — the ideal-$\eta$ $N = 512$ comparison that this note re-runs under realistic $\eta$.
- **`03_in_network_collectives.md`** — SHARP / NVLS effectively dodges contention by doing the reduction in the switch itself; the $\eta$ story for SHARP-enabled AR is much tighter than for software collectives.
- **`references.md`** — NCCL-tests methodology and TPU v4 paper for the calibration sources.
