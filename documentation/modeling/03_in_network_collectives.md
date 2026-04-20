# In-Network Collectives: SHARP, NVLS, and the $N_{\mathrm{hops}}$ Collapse

**Author:** Yue Lu  
**Date:** April 2026  

On switched fabrics — single-switch star (e.g., NVSwitch) or multi-tier star-of-stars (fat-tree / Clos / Dragonfly) — conventional collectives pay an endpoint-driven latency term that grows with group size: $2(N-1)\alpha$ for ring AR, $2\lceil \log_2 N \rceil\alpha$ for the best software tree (DBT / RHD). Every algorithmic step is an endpoint round trip through the switch, and the $\alpha$ floor ($\sim 1\,\mu$s) is set by endpoint software — scheduling, kernel launch, NIC engine setup — paid once per step. For latency-sensitive traffic (small $M$, interactive collectives, fine-grained reductions), this $n_\alpha \cdot \alpha$ term dominates total cost, and no software schedule can eliminate it because the $\alpha$ floor is outside the algorithm's control.

**In-network collectives (INC)** attack this term at its source by moving the reduction into the switch ASIC itself. Switch-resident ALUs reduce flits as they arrive and multicast the result back; endpoints see the reduced output in a single logical round trip, and the endpoint software overhead is paid **once** rather than $O(N)$ or $O(\log N)$ times. On a single-switch star, $n_\alpha$ collapses from $2(N-1)$ or $2\lceil \log_2 N \rceil$ down to $2$, independent of $N$. On a multi-tier switched fabric, the aggregation tree is built from switches rather than endpoints, so $n_\alpha = O(\log_r N)$ but at switch cut-through latency ($\sim$ 200-400 ns) rather than endpoint RTT ($\sim 1-3\,\mu$s) — a 5-10× per-hop saving on top of the structural collapse.

The speedup is **scoped to switched fabrics**: star, fat-tree / Clos, Dragonfly. Torus-native collectives don't benefit — their $N$-dependent $\alpha$ term comes from neighbor-router hops rather than endpoint-driven switch round trips, and there is no switch-hosted ALU in the reduce path. This note focuses on the switched-fabric case: what the hardware does, how the $O(N) \to O(1)$ latency collapse manifests at $N = 512$ on a hypothetical single-switch star (chosen for consistency with the running example in `02_topology_mapping.md §4` and `04_contention_and_congestion.md §6`), and where the mechanism breaks down (reducible types, precision, cross-domain fallback). Three shipping implementations anchor the discussion — NVLS (NVLink SHARP on NVSwitch, single-switch star), Quantum SHARP (InfiniBand, multi-tier), and Tomahawk Ultra INC (Ethernet, emerging).

# Table of Contents

1. [How INC speeds up collectives](#1-how-inc-speeds-up-collectives)
   - 1.1 [Mechanism](#11-mechanism)
   - 1.2 [α-term savings](#12-α-term-savings)
   - 1.3 [BW-term savings](#13-bw-term-savings)
2. [Scale-up vs scale-out](#2-scale-up-vs-scale-out)
   - 2.1 [Scale-up: single-switch star](#21-scale-up-single-switch-star)
   - 2.2 [Scale-out: multi-tier aggregation tree](#22-scale-out-multi-tier-aggregation-tree)
3. [Worked example at $N = 512$](#3-worked-example-at-n--512)
   - 3.1 [The full $N = 512$ ladder (AR)](#31-the-full-n--512-ladder-ar)
   - 3.2 [Regime sensitivity (AR)](#32-regime-sensitivity-ar)
   - 3.3 [Summary (AR)](#33-summary-ar)
   - 3.4 [Cross-primitive comparison (AG / RS / A2A)](#34-cross-primitive-comparison-ag--rs--a2a)
4. [Summary and limitations](#4-summary-and-limitations)
   - 4.1 [Cost summary with INC](#41-cost-summary-with-inc)
   - 4.2 [Limits](#42-limits)
5. [Further reading](#further-reading)

---

## 1. How INC speeds up collectives

### 1.1 Mechanism

Conventional software collectives treat the switch as a passive forwarder: each algorithmic step is an endpoint-to-endpoint message routed through the switch's crossbar without modification. Every step pays one endpoint-software $\alpha$ (NCCL scheduling + CUDA launch + NIC engine setup, $\sim 1\,\mu$s) plus switch cut-through ($\sim 200$ ns) plus wire propagation (negligible at nanoseconds). For AR on $N$ endpoints: ring needs $2(N-1)$ such steps, DBT / RHD needs $2\lceil \log_2 N \rceil$ (§3 of `01_collective_algorithms.md`). The $\alpha$ floor is software-set and cannot be reduced by any choice of algorithm.

```
Software ring AR on a star — every arrow is an endpoint → switch → endpoint round trip,
with one α worth of endpoint software overhead paid at each step

  R0 → switch → R1 → switch → R2 → switch → ... → R(N-1) → switch → R0
  [N-1 RS steps]                                 then [N-1 AG steps]
```

INC replaces the passive-forwarder assumption with an active-switch one. The switch ASIC has ALUs in its crossbar fabric alongside the port-routing logic — they compute sum / max / min / bit-op on incoming flits at line rate and emit a single reduced output. That output is multicast back to all participating ports in one broadcast operation. Endpoints push their contribution once, wait, receive the result once — no intermediate values, no sequential algorithmic steps.

```
INC AR on a star — one logical round trip through the switch ALU

  All endpoints push M bytes → switch ALU reduces in-fabric → switch multicasts M bytes back
  [1 upstream aggregate-reduce + 1 downstream multicast]
  Endpoint software overhead paid once for the entire collective
```

Three structural changes follow, and the next three subsections unpack each in turn: (§1.2) the count of $\alpha$-paying steps collapses; (§1.3) the fabric sees fewer in-flight $M$-byte payloads; (§1.4) the speedup factor differs across collective primitives depending on whether they fit the switch's aggregate-and-fan-out pattern.

### 1.2 α-term savings

Two distinct $\alpha$-term wins come from switch-hosted reduction, and they compound.

**Structural collapse of $n_\alpha$.** On a single-switch star, the entire AR is one logical round trip, regardless of $N$:

$$n_\alpha^{\mathrm{INC}} = 2, \qquad n_\alpha^{\mathrm{ring}} = 2(N-1), \qquad n_\alpha^{\mathrm{DBT}} = 2\lceil \log_2 N \rceil$$

The compression factor is $(N-1)$ vs ring and $\lceil \log_2 N \rceil$ vs DBT. At $N = 72$: $142 \to 2$ (71×) vs ring, $14 \to 2$ (7×) vs DBT. On a multi-tier switched fabric, the INC aggregation tree has depth $k = \lceil \log_r N \rceil$ where $r$ is the per-switch radix, so $n_\alpha = 2k$ — still $O(\log N)$, but counted in switch levels rather than endpoint round trips.

**Per-hop $\alpha$ substitution.** Even when $n_\alpha$ stays $O(\log N)$ on multi-tier fabrics, each hop is cheaper. Software AR's $\alpha$ is endpoint RTT ($\sim 1-3\,\mu$s), dominated by scheduling / kernel launch / NIC engine setup. INC's $\alpha$ for in-fabric levels is switch cut-through ($\sim 200-400$ ns), with the endpoint software overhead paid **once** at collective launch rather than once per level. This is a 5-10× per-hop saving stacked on top of the structural collapse.

The two effects compound. On a single-switch star, the structural collapse dominates and delivers $\sim 70\times$ at modest $N$. On a multi-tier fat-tree across $N = 4096$ ranks, both effects apply: ring AR would need $8190$ endpoint-driven $\alpha$s ($\sim 8$ ms at $\alpha = 1\,\mu$s); INC AR over a 3-level aggregation tree runs at $\sim 2-3\,\mu$s — a $\sim 3000\times$ speedup, of which the structural collapse contributes a factor of $\sim 1000$ and the per-hop $\alpha$ substitution contributes a further $\sim 3$.

**Generalization to other primitives.** The $n_\alpha$ collapse applies to every collective whose dataflow matches switch-hosted aggregate or multicast:

- **AG / Broadcast** — pure replication. The switch multicasts one source (BC) or $N$ per-rank slices (AG) out to all ports in a single switch-local operation. $n_\alpha$ collapses from $(N-1)$ or $\lceil\log_2 N\rceil$ down to $1$ (BC) or $\sim 1$ (AG — a single scatter-gather-multicast operation rather than per-rank multicasts).
- **RS** — fan-in reduction followed by scatter. Switch ALUs reduce all contributions, then the switch distributes slice $k$ to rank $k$. $n_\alpha = 2$ (one aggregate step + one scatter step), same structural collapse as AR but without AR's downstream multicast.
- **A2A** — no match. Every source-destination pair carries a **different** payload, so neither aggregation nor multicast helps. In software, the shipped pairwise A2A needs $(N-1)\alpha$ (one message per destination). A hardware A2A primitive (crossbar scatter-gather: each endpoint submits one combined buffer, switch routes it per the A2A permutation) can collapse this to $\sim 1\alpha$ — but this is an **efficiency** win (avoiding per-round endpoint scheduling), not the same structural $O(N) \to O(1)$ reduction-tree collapse that AR / RS / AG get. Rubin-generation NVSwitches and Tomahawk Ultra both advertise hardware A2A on this basis; NVSwitch Gen4 (current NVL72) does not include it, and A2A on NVL72 runs software-scheduled through the crossbar.

### 1.3 BW-term savings

One structural effect drives the BW-term win: the switch ALU reduces $N$ distinct $M$-byte inputs into **one** $M$-byte output in hardware, so each endpoint handles only $M$ bytes per completed AR instead of the $\sim 2M$ that every software schedule requires. This is the **payload-count collapse** — the BW-side analog of the $n_\alpha$ collapse on the $\alpha$ side, and the entire reason INC keeps winning at large $M$.

**Structural collapse of $\mathrm{BW_{eff}}$.** Define the effective bandwidth as the raw link BW scaled down by how many link-bytes each rank must move to complete one $M$-byte collective:

$$\mathrm{BW_{eff}} \;\equiv\; \mathrm{BW} \cdot \frac{M}{\text{bytes per rank}}, \qquad t_{\mathrm{BW}} = \frac{M}{\mathrm{BW_{eff}}}$$

An algorithm that wastes link BW on duplicate traffic has $\mathrm{BW_{eff}} < \mathrm{BW}$. On a star of $N$ endpoints, the three algorithms land at:

$$\mathrm{BW_{eff}}^{\mathrm{ring}} = \frac{\mathrm{BW}}{2(N-1)/N} \;\longrightarrow\; \frac{\mathrm{BW}}{2}, \qquad \mathrm{BW_{eff}}^{\mathrm{DBT}} = \frac{\mathrm{BW}}{2}, \qquad \mathrm{BW_{eff}}^{\mathrm{INC}} = \mathrm{BW}$$

The compression factor is $2\times$, **independent of $N$** — unlike the $n_\alpha$ collapse of §1.2, which scales with $(N-1)$ or $\log_2 N$. INC closes the algorithmic-vs-link BW gap entirely: the endpoint moves exactly $M$ bytes to complete a collective of size $M$, and the link delivers at its raw rate.

**Why ring / tree cap at $\mathrm{BW}/2$.** Every software AR reduces *at the endpoints*, so every byte of the final result must visit an endpoint link **twice** — once on the way into the reduction, once on the way back out. Ring makes this explicit: during RS each rank forwards $(N-1)/N \cdot M$ bytes of incoming partials to its neighbor while its own data flows out the other direction, then AG re-circulates the reduced result through the same links. RHD / DBT schedule the same two-touch pattern in fewer steps (tree depth $\log_2 N$ instead of $N-1$) but cannot shrink the per-rank byte volume: the reduced result still has to travel up through the tree and back down. NCCL's DBT in fact moves the **full** $2M$ per rank (each rank both receives and forwards the complete final value), so it hits $\mathrm{BW_{eff}} = \mathrm{BW}/2$ exactly, not asymptotically. The factor of two is fundamental to any passive-switch algorithm.

**How INC recovers the full link.** The switch ALU combines all $N$ contributions in-fabric, then multicasts one $M$-byte result. The endpoint pushes $M$ upstream (once) and receives $M$ downstream (once) — no forwarding of peer partials, no "second touch." The two transfers land on opposite directions of the full-duplex link concurrently, so the wall-clock BW term is $M / \mathrm{BW}$, matching the raw link rate. In α-β form:

$$t_{\mathrm{INC, AR}} \;=\; 2\alpha + \frac{M}{\mathrm{BW_{eff}}^{\mathrm{INC}}} \;=\; 2\alpha + \frac{M}{\mathrm{BW}} \qquad \text{vs.} \qquad t_{\mathrm{ring, AR}} \;\longrightarrow\; 2(N-1)\alpha + \frac{2M}{\mathrm{BW}}$$

The $2\times$ BW-term ratio is the **algorithmic ceiling** from the payload-count collapse. The *realized* lift is smaller — switch ALU throughput, multicast contention, and NCCL scheduling overhead all eat into the ceiling — and is treated quantitatively in [`04_contention_and_congestion.md`](04_contention_and_congestion.md).

**Generalization to other primitives.** Not every collective benefits from the payload-count collapse:

- **AG / Broadcast** — **no BW ceiling lift.** Software ring AG on a full-duplex star already runs at $\mathrm{BW_{eff}} = \mathrm{BW}$: each step, a rank forwards $M/N$ outbound while receiving $M/N$ inbound concurrently, and after $N-1$ steps the per-rank wall-clock BW term is $(N-1)/N \cdot M/\mathrm{BW} \approx M/\mathrm{BW}$. Pipelined tree BC matches this. INC's switch-multicast (single push, single broadcast) matches the $M/\mathrm{BW}$ ceiling but doesn't exceed it — the two-touch pattern that costs AR a factor of two never existed for AG / BC. INC AG / BC's entire win is the $n_\alpha$ collapse of §1.2 (from $\lceil\log_2 N\rceil$ or $N-1$ down to 2), so these primitives are dramatic wins at small $M$ and converge toward software at large $M$.
- **RS** — no BW ceiling lift, symmetric argument. Software ring RS already hits $\mathrm{BW_{eff}} = \mathrm{BW}$: the RS phase of ring AR moves $(N-1)/N \cdot M$ per rank in full duplex. INC RS matches the same $(N-1)/N \cdot M/\mathrm{BW}$ ceiling — after the switch reduces, each rank receives only its $M/N$-byte slice — so per-rank traffic is identical to software. INC RS's entire win is on the $\alpha$ side.
- **A2A** — no reduction, no replication, no payload-count collapse. Every source-destination pair carries a **distinct** $M/N$-byte payload; the fabric cannot combine or replicate them, so the total cross-sectional traffic is fixed at $(N-1)\,M$ regardless of schedule. Two further implications specific to A2A:
    1. **Endpoint links are already saturated in software.** Each rank must talk to $N-1$ peers, so a well-scheduled software A2A (e.g., pairing rank $i$ with rank $i \oplus r$ at round $r$) already drives all endpoint-facing ports concurrently. There is no idle-link fan-out for INC to recover.
    2. **Bisection is the bound.** On a single-switch star, every pair's payload crosses the switch crossbar, so the wall-clock BW term is $(N-1)/N \cdot M / \mathrm{BW}$, same as the crossbar aggregate bisection. INC does not move this — a switch ALU cannot reduce data it needs to forward verbatim.

Net effect: **AR is the unique primitive with a BW-side INC win** (the $\mathrm{BW_{eff}}$ doubling from $\mathrm{BW}/2 \to \mathrm{BW}$). AG / BC / RS get only the $\alpha$-side collapse — software already achieves the link-BW ceiling for these primitives, so INC's multicast doesn't lift it further. A2A gets the $\alpha$-side win only if the switch exposes a hardware A2A primitive (see §1.2 — shipping on Tomahawk Ultra, advertised for Rubin-generation NVSwitches, absent on current NVSwitch Gen4 so A2A on NVL72 pays software $n_\alpha$). This asymmetry — only AR benefits structurally in both dimensions; AG / BC / RS / A2A get $\alpha$-side wins of varying size — is why AR is where the INC speedup story lives at large $M$, and why the AG / BC / RS / A2A speedups converge toward software as $M$ grows.

---

## 2. Scale-up vs scale-out

The INC mechanism — switch ASIC reduces in-fabric, endpoints see one round trip — applies at two very different deployment scales. Within one switch domain (**scale-up**), $n_\alpha = 2$ regardless of $N$. Across a multi-tier switched fabric (**scale-out**), $n_\alpha = 2k$ where $k = \lceil\log_r N\rceil$ is the aggregation-tree depth, but each hop is switch cut-through rather than endpoint RTT. Each scale category has two shipping implementations — one NVLink / InfiniBand (NVIDIA) and one Ethernet (Broadcom / NVIDIA).

### 2.1 Scale-up: single-switch star

The entire AR runs inside one switch domain, so $n_\alpha = 2$ end-to-end, independent of $N$ up to the switch radix cap:

$$t_{\mathrm{INC, AR}}^{\mathrm{scale-up}} \;\approx\; 2\alpha_{\mathrm{switch}} + \frac{M}{\mathrm{BW}}$$

per the §1.3 derivation ($\mathrm{BW_{eff}} = \mathrm{BW}$, the full algorithmic ceiling). Two hardware features compose: **hardware multicast** (a single write replicated by the switch to all destination ports in one switch-local operation) and **hardware all-reduce** (switch ALU combines contributions across the SHARP group and multicasts the reduced result back).

Two shipping implementations on different fabrics:

- **NVLS — NVLink / NVSwitch.** NVIDIA's NVLink SHARP, introduced with NVSwitch Gen3 (H100 era) and extended in NVSwitch Gen4 (B200 / GB200 / NVL72). Domain size is capped by NVSwitch radix — 72 GPUs per NVL72 pod. $\alpha_{\mathrm{switch}} \approx 100$–$200$ ns. Current gen supports AR / BC / AG; cross-pod traffic falls back to software (NCCL picks DBT or ring per message size). Rubin-generation NVSwitches are expected to extend NVLS to A2A.
- **Tomahawk Ultra INC — Ethernet.** Broadcom Tomahawk Ultra (shipped 2025) is the first Ethernet switch ASIC with in-network collectives [`[TH-ULTRA]`]. 51.2 Tbps full-duplex, $\alpha_{\mathrm{switch}} \approx 250$ ns. Structurally equivalent to NVLS — single-switch star with in-fabric reduction — but on commodity Ethernet, breaking the NVLink-only monopoly on scale-up INC and opening the door to INC-enabled Ethernet fabrics at HPC / AI scale-up cost points.

### 2.2 Scale-out: multi-tier aggregation tree

Across thousands of endpoints spread over many switch levels, the INC primitive is an **aggregation tree** whose internal nodes are switch ASICs. Each level reduces its incoming flits and forwards the $M$-byte partial upward; the root completes the reduction and multicasts back down:

$$t_{\mathrm{INC, AR}}^{\mathrm{scale-out}} \;\approx\; 2k \cdot \alpha_{\mathrm{switch}} + \frac{M}{\mathrm{BW}}$$

where $k$ is the **number of switch tiers** an $M$-byte flit traverses from an endpoint to the root of the aggregation tree. For a fat-tree with per-switch radix $r$ (fan-in per level), covering $N$ endpoints requires $k = \lceil\log_r N\rceil$ tiers — e.g., $r = 64$ ports per switch covers $N = 4096$ endpoints in $k = 2$ tiers (leaf + spine), or $N = 262{,}144$ in $k = 3$ (leaf + spine + super-spine). The factor of $2k$ on the $\alpha$ term accounts for the round trip: $k$ switch hops up to the root during reduce, then $k$ back down during multicast. $\alpha_{\mathrm{switch}} \approx 200$–$400$ ns is the switch cut-through, not endpoint software RTT. The BW term stays at $M/\mathrm{BW}$ — the same algorithmic ceiling as scale-up ($\mathrm{BW_{eff}} = \mathrm{BW}$ from §1.3), because each endpoint still only pushes $M$ up and receives $M$ down, and cut-through pipelining at each tier keeps the endpoint's outbound and inbound directions overlapping once the pipeline is filled ($\sim 2k\alpha_\mathrm{switch}$).

Concretely, for $N = 4096$ across a 3-tier aggregation tree, AR latency is $\sim 2$–$3\,\mu$s — even though software ring AR at the same $N$ would need 8190 sequential endpoint $\alpha$s ($\sim 8$ ms at endpoint $\alpha = 1\,\mu$s). The $\sim 3000\times$ speedup is dominated by the endpoint-to-switch $\alpha$ substitution combined with the structural $O(N) \to O(\log N)$ collapse.

Two shipping implementations on different fabrics:

- **Quantum SHARP — InfiniBand.** NVIDIA Mellanox Quantum-2 and Quantum-X800 switches on IB fat-tree / Clos topologies. The established scale-out INC path for large training clusters.
- **Spectrum-X SHARP — Ethernet.** NVIDIA's Ethernet-side analog, running the SHARP protocol over RoCE on Spectrum-X switches. Brings the scale-out INC story to commodity-Ethernet clusters — complementing Tomahawk Ultra on the scale-up side.

---

## 3. Worked example at $N = 512$

To track the same $N = 512$ anchor used in `02_topology_mapping.md §4` and `04_contention_and_congestion.md §6`, apply scale-up INC to a **hypothetical single-switch star with 512 ports**. Real scale-up INC (NVL72) caps at $N = 72$ — a 512-port single-switch INC ASIC does not exist today, so a production $N = 512$ deployment would use scale-out INC ($k = 2$ tiers) per §2.2. We pick the single-switch abstraction for the worked example below because it applies the §1.2 / §1.3 algorithmic ceilings directly ($n_\alpha = 2$, $\mathrm{BW_{eff}} = \mathrm{BW}$); scale-out INC at $k = 2$, $\alpha_\mathrm{switch} = 0.5\,\mu$s gives essentially the same total ($\sim 20\,\mu$s vs $\sim 19\,\mu$s for scale-up) because the $M / \mathrm{BW}$ term dominates at $M = 16\,\mathrm{MB}$.

Per-link $\alpha = 0.5\,\mu$s (switch cut-through + endpoint software), $\mathrm{BW} = 900\,\mathrm{GB/s}$, $M = 16\,\mathrm{MB}$ — same parameters as `02 §4.1`.

### 3.1 The full $N = 512$ ladder (AR)

Copying the three software / torus rows from `02 §4.1` verbatim and adding one **NVLS-style INC** row on the hypothetical single-switch star:

| Topology | Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| Star (crossbar) | Ring AR | 1022 | 511 μs | 35.5 μs | **546 μs** |
| Star (crossbar) | DBT / RHD | 18 | 9 μs | 35.5 μs | **45 μs** |
| Hypothetical 512-port star | **NVLS-style INC** | 2 | 1 μs | 17.8 μs | **18.8 μs** |
| Torus $8 \times 8 \times 8$ | Dim-decomp ring | 42 | 21 μs | 35.5 μs | **57 μs** |

INC closes two gaps at once:

- **$\alpha$ side.** $n_\alpha$ collapses from 18 (DBT) or 1022 (ring) to 2. The α term shrinks from 9 μs (DBT) to 1 μs (INC) — small in absolute terms at this $M$, but consequential at smaller $M$ (see §3.2).
- **BW side.** $\mathrm{BW_{eff}}$ doubles from $\mathrm{BW}/2$ (any software schedule) to $\mathrm{BW}$ (INC), halving the BW term from 35.5 μs to 17.8 μs.

Net: star + INC at **18.8 μs** beats the best software pairing (star + DBT at 45 μs) by $\sim 2.4\times$ and the torus at 57 μs by $\sim 3\times$. Against the pathological star+ring row (546 μs) the speedup is $\sim 29\times$, but that row is a cautionary baseline, not a design we'd ship. The $\sim 2\times$ ceiling from §1.3 is the asymptotic BW-side INC speedup; the full $\sim 2.4\times$ vs DBT at $M = 16\,\mathrm{MB}$ reflects the residual $\alpha$-side contribution.

### 3.2 Regime sensitivity (AR)

The $N = 512$ speedup of INC over the best software pairing (DBT) varies sharply with $M$. Sweeping $M$ from $\alpha$-bound to BW-bound:

| $M$ | Star + DBT | Star + INC | Speedup (DBT → INC) |
|---|---|---|---|
| 10 KB ($\alpha$-bound) | 9.02 μs | 1.01 μs | **~9×** |
| 1 MB | 11.2 μs | 2.1 μs | **~5×** |
| 16 MB (anchor) | 45 μs | 18.8 μs | **~2.4×** |
| 1 GB (BW-bound) | 2.23 ms | 1.11 ms | **~2×** |

At small $M$ the $\alpha$-term collapse dominates: DBT still needs 18 synchronizations at 0.5 μs each (= 9 μs); INC needs 2 (= 1 μs). At large $M$ the BW-term ratio takes over and the speedup converges to the $2\times$ payload-count ceiling. The transition is governed entirely by the $\alpha$-vs-BW crossover for DBT: $M^\star \approx n_\alpha^{\mathrm{DBT}} \cdot \alpha \cdot \mathrm{BW} / 2 = 18 \cdot 0.5\,\mu\mathrm{s} \cdot 900\,\mathrm{GB/s} / 2 \approx 4\,\mathrm{MB}$. Below $M^\star$ INC's $\alpha$ savings dominate; above, its BW savings do.

### 3.3 Summary (AR)

At the $N = 512$, $M = 16\,\mathrm{MB}$ anchor:

- **Star + INC** = 18.8 μs — algorithmic-ceiling best.
- **Star + DBT** = 45 μs — software optimum on the same star, $2.4\times$ slower than INC.
- **Torus $8^3$ dim-decomp** = 57 μs — best non-INC option, $3\times$ slower than INC.
- Star + ring = 546 μs — pathological pairing, shown for contrast.

**Regime takeaway.** INC's $\alpha$-term collapse of §1.2 dominates the DBT-vs-INC gap for $M \lesssim 4\,\mathrm{MB}$; above that the $\mathrm{BW_{eff}}$ doubling of §1.3 takes over and the speedup converges to $\sim 2\times$. Both wins compose — INC always beats ring, DBT, and torus — but the attribution shifts with $M$.

The ceilings in this section assume frictionless cut-through, no ALU or multicast contention, and no scheduler overhead. `04_contention_and_congestion.md §6` re-runs the same $N = 512$ ladder under realistic $\eta$ and quantifies how much of each ceiling survives in deployment.

### 3.4 Cross-primitive comparison (AG / RS / A2A)

§3.1–§3.3 focused on AR. The $\alpha$-side INC collapse applies to AG, BC, and RS as well; the BW-side collapse is AR-exclusive (§1.3 — AG / RS already hit $\mathrm{BW_{eff}} = \mathrm{BW}$ in software via full-duplex operation). A2A gets no INC lift on current shipping hardware (NVSwitch Gen4 ships no hardware A2A primitive). The table runs the same $N = 512$, $M = 16\,\mathrm{MB}$, $\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$ anchor across all three non-AR primitives:

| Primitive | Topology / Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| **AG / RS** | Star ring | 511 | 255.5 μs | 17.7 μs | **273 μs** |
|  | Star RHD (pipelined tree) | 9 | 4.5 μs | 17.7 μs | **22.2 μs** |
|  | Hypothetical 512-port star + INC | 2 | 1.0 μs | 17.7 μs | **18.7 μs** |
|  | Torus $8 \times 8 \times 8$ dim-decomp | 21 | 10.5 μs | 17.7 μs | **28.2 μs** |
| **A2A** | Star pairwise (NCCL) | 511 | 255.5 μs | 17.7 μs | **273 μs** |
|  | Hypothetical 512-port star + INC | — | — | — | **N/A on shipping hardware** |
|  | Torus $8 \times 8 \times 8$ bisection-bound (TPU / Trainium) | 12 | 6 μs | 35.5 μs | **41.5 μs** |

Three observations:

1. **AG / RS INC closes the α-side gap but not the BW-side gap.** Star + INC (18.7 μs) beats star RHD (22.2 μs) by only $\sim 1.2\times$ at this anchor — a much tighter margin than AR's $2.4\times$ at the same $M$. The entire gap is the $\alpha$-term collapse ($4.5 \to 1.0\,\mu$s); both rows share the same 17.7 μs BW term because software RHD already hits $\mathrm{BW_{eff}} = \mathrm{BW}$ on a full-duplex star. At small $M$ the ratio widens toward the $\alpha$-only ceiling ($\sim 4.5\times$); at large $M$ AG / RS INC speedup collapses to $1\times$.
2. **A2A gets no INC lift on shipping hardware.** NVSwitch Gen4 includes no hardware A2A primitive, so A2A on NVL72 falls back to software-scheduled crossbar traffic. Even with the HW-A2A primitive on Tomahawk Ultra or Rubin-generation NVSwitches, the BW term would stay at the bisection bound — a switch ALU cannot reduce data it needs to forward verbatim. A2A's win path on shipping hardware stays topology-side: star pairwise pays the full $(N-1)\alpha$; torus bisection-bound reduces it to $\mathrm{diam}\cdot\alpha$ while paying the $D_{\max}/4$ BW penalty.
3. **Torus stays competitive for AG / RS / A2A when INC is unavailable.** At $N = 512$, $M = 16\,\mathrm{MB}$, torus 8³ AG / RS (28.2 μs) is $\sim 1.5\times$ slower than hypothetical INC but $\sim 10\times$ faster than star ring — the dim-decomposition's $\sum(D_i - 1)$ $\alpha$ collapse carries most of the win. Torus is the best A2A option by a wide margin when INC is unavailable (41.5 μs vs star pairwise's 273 μs), because the dim-decomposed $\mathrm{diam}\cdot\alpha$ latency term cuts deeper than the $D_{\max}/4$ BW penalty adds.

The pattern across all four primitives: **AR is the primitive where INC pays back at every $M$** (both $\alpha$ and BW wins); AG / RS wins concentrate at small $M$; A2A gets no INC win at all on shipping hardware. `04 §6` layers realistic $\eta$ on each of these rows and quantifies which speedups survive contention.

---

## 4. Summary and limitations

### 4.1 Cost summary with INC

The table below echoes `02_topology_mapping.md` §5.1 (star / torus merged rows by topology) with an INC row added under **Star** showing the $\alpha$ collapse and the $\mathrm{BW_{eff}}$ doubling. The last column quotes the **algorithmic-ceiling INC speedup** against each baseline — for star rows, the speedup of INC-on-star over the software algorithm on the same star; for torus rows, the speedup of switching the deployment to an INC-enabled star entirely (INC itself is not implementable on torus, but the architectural comparison is meaningful).

| Topology | Primitive | Algorithm | Latency term | BW term | INC speedup (small $M$ / large $M$) |
|---|---|---|---|---|---|
| **Star** | AR | Ring | $2(N-1)\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)\times$ / $\sim 2\times$ |
|  | AR | DBT | $2\lceil\log_2 N\rceil\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim \log_2 N\times$ / $\sim 2\times$ |
|  | AR | **NVLS (INC)** | $2\alpha$ | $M/\mathrm{BW}$ | — (reference: INC itself) |
|  | AG / BC | Ring AG / pipelined BC | $(N-1)\,\alpha$ / $\lceil\log_2 N\rceil\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)\times$ or $\sim \log_2 N\times$ / $1\times$ ($\alpha$-only) |
|  | AG / BC | **Multicast engine (INC)** | $2\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | — (reference: INC itself) |
|  | RS | Ring RS | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $\sim (N-1)\times$ / $1\times$ ($\alpha$-only) |
|  | RS | **INC reduce-then-scatter** | $2\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | — (reference: INC itself) |
|  | A2A | Pairwise | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | $1\times$ / $1\times$ (no HW-A2A on NVSwitch Gen4) |
| **Torus** | AR | Dim-decomp ring | $2\sum_i (D_i-1)\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | $\sim \sum_i (D_i-1)\times$ / $\sim 2\times$ (architectural swap) |
|  | A2A | Pairwise — bisection-bound | $\mathrm{diam}\cdot\alpha$ | $D_{\max}/4 \cdot M/\mathrm{BW}$ | $1\times$ / $1\times$ (A2A doesn't compose with INC) |

Three observations:

1. **INC is implemented on star, but its speedup ceiling applies across topology choices.** The primitive itself lives inside a switch ASIC, so torus-native fabrics have no direct INC path — the $N$-dependent $\alpha$ on torus comes from neighbor-router hops, which cannot be collapsed by a central switch that does not exist. However, the algorithmic-ceiling ratios above compare INC-on-star to each baseline (star ring, star DBT, torus dim-decomp ring) on equal footing — showing, for example, that replacing a 3D torus AR with an INC-enabled star delivers the same $\sim \sum_i(D_i-1) \times$ $\alpha$-side win as replacing star-ring with INC. This is the asymmetric architectural improvement flagged in `02_topology_mapping.md` §5.1 observation 4.
2. **Latency-regime speedup is massive for every primitive; BW-regime speedup applies to AR only.** INC compresses $n_\alpha$ by a factor of $(N-1)$ (vs ring) or $\log_2 N$ (vs DBT) across AR, AG, BC, and RS — tens to hundreds at production $N$. The BW-regime gain is **AR-exclusive** and capped at $\sim 2\times$ (the payload-count collapse's theoretical upper bound; $\sim 1.3\times$ measured on NVLink, $\sim 1.5\text{–}1.7\times$ on IB). AG / BC / RS converge to $1\times$ at large $M$ because software already achieves $\mathrm{BW_{eff}} = \mathrm{BW}$ on a full-duplex star. Latency-sensitive workloads benefit most across the board; AR alone keeps winning at bulk sizes.
3. **A2A does not compose with INC on shipping hardware.** A2A has no aggregation or replication step for the switch ALU to exploit; it is all point-to-point permutations. Tomahawk Ultra and Rubin-generation NVSwitches advertise a hardware A2A primitive that collapses the $\alpha$-side per-destination scheduling to $\sim 1\alpha$, but current shipping NVSwitch Gen4 (NVL72) does not include it — A2A on NVL72 runs software-scheduled through the crossbar. Even where HW-A2A ships, the BW term stays at the bisection bound: a switch ALU cannot reduce data it needs to forward verbatim. This keeps A2A the bisection-bound primitive on both torus (§3.3 of `02_topology_mapping.md`) and star.

### 4.2 Limits

Every cost formula above assumes the INC path is available and the collective fits the switch's hardware capabilities. Five restrictions apply in practice.

1. **Scope: single switch domain (or SHARP-enabled hierarchy).** NVLS works only within one NVSwitch domain. Quantum SHARP works across a SHARP-enabled IB fabric, but only if all switches in the aggregation tree support the protocol. Cross-domain traffic falls back to software. In practice this means INC helps TP / EP within an NVL72 but requires explicit multi-domain SHARP configuration for scale-out; otherwise a cross-domain deployment pays INC cost within each domain + software cost across.
2. **Reducible types: limited.** Hardware ALUs support sum, max, min, bit-and/or, and a handful of others. Softmax-normalization reductions, top-k, and similar complex reductions must stay in software.
3. **Precision: hardware-fixed.** Switch ALUs typically operate in BF16 with FP32 accumulators; some SHARP-enabled IB switches support FP32 directly. Users relying on FP64 reductions (rare in modern ML, common in scientific simulation) should verify SHARP support on their switch gen. The accumulation order is also fixed by the switch tree shape, which can produce slightly different numerical results than ring AR — usually within numerical tolerance but worth flagging for bitwise-reproducibility use cases.
4. **Message alignment.** INC operations require specific alignment (typically 128 B or higher). Very small messages can fall through to software — NCCL's heuristic handles this, but a sweep across $M$ may show a sudden floor where the runtime switches paths.
5. **BW-regime convergence.** INC's dominant win is in the latency-term regime. At large $M$, ring AR's bandwidth-optimal bound $2(N-1)/N \cdot M/\mathrm{BW}$ matches INC's $M/\mathrm{BW}$ modulo the 2× BW-eff accounting. The 70× speedup at small $M$ shrinks to $\sim 2\times$ at large $M$ — enough to matter for bulk gradient / parameter transfers, not enough to change the qualitative cost model.
6. **Ceilings vs realized speedups.** The ratios in §4.1 and the numbers in §3 are **algorithmic ceilings** — they assume frictionless cut-through pipelining, no multicast contention, and no NCCL / switch-ALU scheduling overhead. Realized speedups on production hardware are smaller, particularly in the BW regime (e.g., NVLink NVLS measures $\sim 1.3\times$ BW-regime lift vs the $2\times$ ceiling). The full contention-coefficient treatment — calibrating $\eta_\alpha, \eta_\beta$ against published busbw measurements and re-running the comparison under realistic factors — is in [`04_contention_and_congestion.md`](04_contention_and_congestion.md).

---

## Further reading

- **`01_collective_algorithms.md`** — baseline ring and tree AR costs that INC compresses against; the $\alpha$-$\beta$ model and the Rabenseifner / DBT derivations referenced throughout.
- **`02_topology_mapping.md` §2** — $\alpha$ / BW calibration on a scale-up star and star-specific observations (e.g., DBT's port-uniform utilization on the crossbar). §5.1 observation 4 flags INC as star's $\alpha$-compression escape hatch.
- **`04_contention_and_congestion.md`** — contention coefficients that modify software collectives; INC paths are less sensitive to contention because they use only one switch operation.
- **`references.md`** — primary sources for SHARP (Graham et al. 2016), NVLink SHARP / NVLS (NVIDIA 2023 whitepapers, NCCL 2.27 release notes), and Tomahawk Ultra INC (Broadcom 2025).
- **Graham et al. (2016)**, "Scalable Hierarchical Aggregation Protocol (SHARP)" — the architectural paper.
- **NVIDIA (2023)**, NVLink SHARP and NVSwitch Multicast/Reduce Whitepapers.
