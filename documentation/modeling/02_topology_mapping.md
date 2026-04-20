# Mapping Collective Algorithms to Physical Topologies

**Author:** Yue Lu  
**Date:** April 2026  

The previous note (`01_collective_algorithms.md`) costs collective primitives on an abstract fully-connected fabric. Real clusters don't have that. They have a star topology of GPUs around an NVSwitch, or a 2D / 3D torus for TPU pods. This note walks through each topology with a 2×2-style diagram, shows how ring / tree / dim-decomposed ring AR maps onto each, derives the cost formulas, and ends with a side-by-side numerical comparison at $N = 512$.

No network contention modeled here — every physical link is assumed to carry the active collective alone at peak bandwidth, with no concurrent flows from other collectives, other groups, or background traffic sharing the link. Realistic link sharing, bisection oversubscription, and concurrent-group conflicts are scored in `04_contention_and_congestion.md`.

# Table of Contents

1. [Topology primer](#1-topology-primer)
2. [Star topology](#2-star-topology)
   - 2.1 [All-reduce](#21-all-reduce)
   - 2.2 [All-gather / reduce-scatter](#22-all-gather--reduce-scatter)
   - 2.3 [All-to-all](#23-all-to-all)
3. [Torus topology](#3-torus-topology)
   - 3.1 [All-reduce — dim-decomposed ring](#31-all-reduce--dim-decomposed-ring)
   - 3.2 [All-gather / reduce-scatter](#32-all-gather--reduce-scatter)
   - 3.3 [All-to-all (bisection-bound)](#33-all-to-all-bisection-bound)
4. [Side-by-side comparison at N = 512](#4-side-by-side-comparison-at-n--512)
   - 4.1 [All-reduce](#41-all-reduce)
   - 4.2 [All-gather / reduce-scatter](#42-all-gather--reduce-scatter)
   - 4.3 [All-to-all](#43-all-to-all)
   - 4.4 [Key observations](#44-key-observations)
5. [Summary and limitations](#5-summary-and-limitations)
6. [Appendix A: Dim-decomposed Rabenseifner halving-doubling](#appendix-a-dim-decomposed-rabenseifner-halving-doubling)
7. [Further reading](#further-reading)

---

## 1. Topology primer

We focus on two topologies that cover the majority of deployed AI and HPC fabrics: **star** (endpoints around a central high-radix switch — e.g., NVSwitch) and **torus** (a $k$-dimensional lattice with wraparound — e.g., TPU pods). Fat-tree / Clos fabrics behave like a multi-level star for the α-β analysis and map onto the star results with minor adjustments, so we don't treat them separately. Four properties describe any topology:

- **Diameter** $D$: longest shortest-path in hops between any two ranks.
- **Bisection bandwidth**: aggregate BW across a worst-case partition cut.
- **Radix**: ports per switch (for switched fabrics) or ports per router (for direct fabrics).
- **Scalability**: how wiring and cost scale with $N$.

### 1.1 Star (single high-radix switch)

All $N$ endpoints attach to one central switch. $D = 2$ link hops (endpoint → switch → endpoint). This is NVLink 5 + NVSwitch Gen4 within a single NVL72 domain.

```
        ┌──────────────────────────────────────┐
        │              NVSwitch                │
        │         (N ports, full crossbar)     │
        └──┬─────┬─────┬─────┬─────┬─────┬────┘
           │     │     │     │     │     │
          R0    R1    R2    R3    ...   RN-1
```

- **Pros:** any-to-any in 2 hops. Collective algorithm is unconstrained by wiring (star can emulate any logical topology — ring, tree, butterfly — because every endpoint is 1 switch hop from every other).
- **Cons:** scale limit is switch radix. A single NVSwitch Gen4 domain caps at ~72 GPUs; beyond that you need a multi-tier fabric.

### 1.2 Torus (2D or 3D lattice with wraparound)

Ranks are laid out on a $k$-dimensional lattice; each rank connects directly to $2k$ neighbors (one per dim per direction), with wraparound edges closing each row into a ring. TPU v4/v5p pods use a 3D torus.

```
2D torus example: 4×4 = 16 ranks

  (0,3)━(1,3)━(2,3)━(3,3) ┐   wraparound Y
    ┃     ┃     ┃     ┃   │
  (0,2)━(1,2)━(2,2)━(3,2) │
    ┃     ┃     ┃     ┃   │
  (0,1)━(1,1)━(2,1)━(3,1) │
    ┃     ┃     ┃     ┃   │
  (0,0)━(1,0)━(2,0)━(3,0) ┘
    └─────────────────────→ wraparound X
     each row is a ring;  each column is a ring
```

- **Pros:** no central switch — each rank has only $2k$ links. Linear wire cost in $N$. Dim structure enables dim-by-dim ring collectives that compress latency from $O(N)$ to $O(N^{1/k})$.
- **Cons:** diameter $D = \sum_i \lfloor D_i / 2 \rfloor$ grows as $N^{1/k}$; bisection scales as $N^{(k-1)/k}$ — sub-linear. All-to-all is bisection-constrained (more below).

---

## 2. Star topology

**Star is the most flexible substrate for mapping collective algorithms to hardware.** Any-to-any in 2 hops means every logical "communication edge" in a ring, butterfly, DBT, or pairwise schedule lands on a single switch cut-through — the algorithm's logical topology *is* the physical topology, with no wiring-induced translation. Any of the seven primitives from [`01_collective_algorithms.md`](01_collective_algorithms.md) runs on a star at its *pure* α-β cost — no topology correction, no dim-decomposition, no hierarchical sub-tiers. **The algorithm is a software choice, not a wiring constraint.** This direct-mapping property is unique to star among the topologies in this series: torus forces dim-decomposition (§3); multi-tier fat-tree / Clos fabrics force tier-aware schedules that sub-divide the collective across per-tier sub-groups. There are no star-specific "special" schedules; this section only records the α calibration that makes the cost formulas concrete on a representative scale-up star, and defers algorithm selection to the prior note. In-network collectives (SHARP, covered in [`03_in_network_collectives.md`](03_in_network_collectives.md)) build on this by collapsing even the tree's $\log_2 N$ endpoint hops to $O(1)$ switch-hop latency — an amplification of star's algorithmic freedom, not a separate algorithm class.

**α and BW calibration on a scale-up star.** On NVLink-5 / NVSwitch Gen4 class hardware:

- $\alpha \approx 0.5\,\mu$s — switch cut-through ($\sim$100 ns) plus endpoint software overhead ($\sim$400 ns). Each algorithm "hop" in the star primitive costs this $\alpha$, which already includes the physical 2-link endpoint→switch→endpoint traversal as a single cut-through event.
- $\mathrm{BW} \approx 900\,\mathrm{GB/s}$ per direction per GPU — a single NVLink-5 port's unidirectional bandwidth. All cost formulas from the prior note use this as the per-port BW.

### 2.1 All-reduce

All four AR algorithms from the prior note (ring in §3.1, DBT in §3.2, plus simple rec-doubling in App. A.1 and Rabenseifner in App. A.2 for reference) run on a star at their pure costs. The full comparison — pipelining / port-budget argument for why NCCL ships ring and DBT, the Rabenseifner caveats, and the ring-vs-DBT and rec-doub-vs-DBT numerical crossovers at these same NVLink-class parameters — lives there in §3.3. NCCL ships both ring and DBT on star and its tuner selects per $(N, M)$: **empirically DBT wins small-$M$ and ring wins large-$M$** [DEMYST-NCCL] — the opposite of the pipelined α-β prediction, which says DBT's bandwidth-optimal pipelining should make it win across the whole range above rec-doubling's sub-MB regime. The inversion traces to implementation-side tree overhead (finite pipeline depth, per-step kernel complexity, empirical BW-coefficient constant above ring's $2(N-1)/N \to 2$ floor) at bulk sizes; see 01 §3.3 for the pure-model derivation and the "practice" caveat on the takeaway line. Rec-doubling remains reserved for sub-MB control messages because its $\log_2 N \cdot M/\mathrm{BW}$ BW term is prohibitive above that, and ring's $(N-1)\alpha$ term is prohibitive below the crossover against DBT. One star-specific note worth flagging: DBT's complementary-role assignment (every rank is interior in one tree and leaf in the other) keeps per-switch-port utilization uniform on the crossbar, avoiding the single-port ceiling that a single-tree root would hit — a centralized-switch advantage that the abstract analysis doesn't visualize but that keeps DBT in the shipping menu on NVSwitch-class fabrics for the small-$M$ regime where it wins.

### 2.2 All-gather / reduce-scatter

Star runs both ring AG/RS (01 §4.1) and rec-doubling AG / rec-halving RS (01 App. B.1) at their pure costs from the prior note. Ring is NCCL's choice despite the $\log_2 N$ α advantage of rec-doubling / rec-halving — the full pipelineability / any-$N$ / kernel-simplicity / compute-overlap argument is in §4.2 there. PAT (01 App. B.2) also ships in NCCL 2.23+ [NCCL-PAT] but only for the **inter-node, 1-rank-per-node regime** — on a pure-star scale-up fabric where all ranks sit inside one NVSwitch domain, PAT is not selected. Again, no star-specific correction to the cost formulas: `BW` = per-port BW, α = cut-through + software, plug in and read off.

### 2.3 All-to-all

A2A on a star is bounded by per-rank port BW, not by the switch fabric. The switch has aggregate BW $N \cdot \mathrm{BW}$ — far more than any collective needs — but each endpoint has only one duplex link, so the information-theoretic lower bound on cost is $\frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}} \approx M/\mathrm{BW}$ (each rank must ship $(N-1)M/N$ bytes through its one port). **Star's win over torus for A2A is that it *achieves* this BW lower bound with no bisection bottleneck and no dim-decomposition to engineer** — just peak per-port BW applied uniformly across all $N(N-1)$ pairs. NCCL ships pairwise direct-send (01 §5.1) on star: it hits the BW bound but pays $(N-1)\alpha$ on the latency side, so the α term dominates the cost on scale-out fabrics where per-hop α is in the μs range (see §5.2 of the prior note for the full rationale, the $(N, M)$ mapping to workload parameters, and the discussion of what the shipped default leaves on the table).

---

## 3. Torus topology

A torus trades star's algorithmic flexibility for linear wire cost at scale, and the price is a more constrained algorithmic menu. You *can* still run any of the prior note's flat algorithms on a torus and get the correct answer — a flat $N$-ring embedding, say — but "neighbor" in the linearized rank order physically crosses many torus links per logical hop on average, so the cost is the full flat-ring formula with none of the torus's product structure exploited. The optimization the torus makes available is to rewrite each primitive dim-by-dim, running independent collectives along each axis; when this applies, it turns the $O(N)$ latency of a flat ring into $O(N^{1/k})$ for a $k$-D torus while keeping the bandwidth term bandwidth-optimal. When it doesn't, the cost collapses back toward flat-ring: the compression only materializes under dim-aligned group layouts (§3.1), A2A is bisection-bound rather than per-port and scales with $D_{\max}$ not $N$ (§3.3), and asymmetric dim sizes or skewed traffic pay disproportionately. The subsections below walk through each primitive's mapping and call out where it breaks; realistic-$\eta$ re-scoring of both wins and losses lives in `04_contention_and_congestion.md`.

### 3.1 All-reduce — dim-decomposed ring

AR on a torus ships in one form on production hardware: **dim-decomposed ring**, which exploits the torus's product-structured wiring to compress the flat-ring α cost from $O(N)$ to $O(N^{1/k})$ at bandwidth-optimal BW. TPU (XLA / JAX) and AWS Trainium (NeuronX CCL) both default to it for every sharded reduction. A tree-flavored alternative — **dim-decomposed Rabenseifner halving-doubling** — swaps the inner primitive of each dim-phase from ring to the power-of-2 halving-doubling schedule for additional α compression; it is *not* a shipping algorithm on any production torus (see Appendix A for the full derivation, cost formula, and production-status discussion) but is included as a reference point that sharpens the production-ring analysis in this subsection. Both share the same correctness argument — reduction's associativity lets the collective factor across dims — and differ only in what runs within each dim-phase.

**Relation to the ring AR in the prior note.** Dim-decomposed ring AR is not a new primitive — it is a topology-aware *composition* of the same ring RS+AG building block derived in the prior note, applied $k$ times along the torus's $k$ physical dims. (Flat-ring AR from 01 still runs correctly on a torus; dim-decomp is the optimization that exploits the product-structured wiring instead of paying the flat-ring $O(N)$-hop cost.) Each dim-$i$ phase runs the identical $D_i$-rank ring RS (or AG) with the identical α-β cost formula, just on a shorter ring. What dim-decomposition adds on top is two things: (i) **serial composition across dims**, valid because reduction is associative and the bandwidth term telescopes as the per-rank chunk size shrinks phase-by-phase, and (ii) **concurrent execution of $(N/D_i)$ independent rings within each phase**, only possible because the torus's $k$-D Cartesian-product wiring gives each dim-ring its own physically disjoint copper — no synchronization, no shared bandwidth. Everything below is the familiar ring-RS/AG cost from the prior note applied $2k$ times with this compositional bookkeeping; the rest of the subsection walks through why it's correct and where the $O(N) \to O(N^{1/k})$ α-compression comes from.

**Algorithm.** Run the Patarasuk-Yuan RS+AG schedule one dim at a time. Phase order on a 3D torus: X-RS, Y-RS, Z-RS, then Z-AG, Y-AG, X-AG — $2k$ phases total. The AG dims traverse in reverse of the RS dims so that chunk sizes grow back symmetrically to how they shrank during RS (the telescoping in the cost derivation below relies on this pairing; correctness does not — any AG dim order finishes AR correctly). Within each dim-$i$ phase, $(N/D_i)$ rings run *concurrently* on the $(N/D_i)$ dim-lines perpendicular to the active axis; each ring exchanges only with its own dim-neighbors over dedicated copper, with no bytes crossing between rings.

**Why it works — 2×2 worked example.** Four ranks arranged in a 2×2 grid, each holding a length-4 input vector. Subscripts denote chunk index 0–3.

```
Initial state — each cell shows what that rank holds:

          x=0                       x=1
       ┌──────────────────────┬──────────────────────┐
  y=1  │  [c₀, c₁, c₂, c₃]   │  [d₀, d₁, d₂, d₃]  │
       ├──────────────────────┼──────────────────────┤
  y=0  │  [a₀, a₁, a₂, a₃]   │  [b₀, b₁, b₂, b₃]  │
       └──────────────────────┴──────────────────────┘
```

*Phase 1 (X-dim RS, two rings in parallel):* row $y=0$ runs its 2-rank ring on $\{a, b\}$; row $y=1$ runs its 2-rank ring on $\{c, d\}$. No data crosses between rows — the two physical X-axes are independent copper. Each row sends half its vector, receives the other half, and adds.

```
          x=0                            x=1
       ┌────────────────────────┬────────────────────────┐
  y=1  │  [c₀+d₀, c₁+d₁]        │  [c₂+d₂, c₃+d₃]       │  ← row 1 ring output
       ├────────────────────────┼────────────────────────┤
  y=0  │  [a₀+b₀, a₁+b₁]        │  [a₂+b₂, a₃+b₃]       │  ← row 0 ring output
       └────────────────────────┴────────────────────────┘
```

Row 0's partial sums touch only $\{a, b\}$; row 1's touch only $\{c, d\}$. Chunk size has halved. Row 0 did not wait for row 1, nor vice versa — embarrassingly parallel, no coordination.

*Phase 2 (Y-dim RS, two rings in parallel):* column $x=0$ runs on $\{[a_0+b_0, a_1+b_1], [c_0+d_0, c_1+d_1]\}$; column $x=1$ similarly. Each column's ring treats its inputs as ordinary numbers — it doesn't know (and doesn't need to know) that these are already partial sums from phase 1.

```
          x=0                              x=1
       ┌───────────────────────────┬───────────────────────────┐
  y=1  │  [a₁+b₁+c₁+d₁]            │  [a₃+b₃+c₃+d₃]           │
       ├───────────────────────────┼───────────────────────────┤
  y=0  │  [a₀+b₀+c₀+d₀]            │  [a₂+b₂+c₂+d₂]           │
       └───────────────────────────┴───────────────────────────┘
```

Exactly the RS target — each rank holds one fully-reduced chunk of size $M/N = M/4$.

*Phase 3 (Y-dim AG, two rings in parallel):* Y is the last dim RS touched, so AG starts there. Column $x=0$'s 2-rank ring exchanges its single chunks — $(0,0)$ sends chunk 0 to $(0,1)$ and receives chunk 1; column $x=1$'s ring exchanges chunk 2 ↔ chunk 3. No reduction, just copies. After this phase each rank holds two chunks, chunk size back up to $M/D_x = M/2$.

```
          x=0                                    x=1
       ┌───────────────────────────────────┬───────────────────────────────────┐
  y=1  │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁]        │ [a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]        │
       ├───────────────────────────────────┼───────────────────────────────────┤
  y=0  │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁]        │ [a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]        │
       └───────────────────────────────────┴───────────────────────────────────┘
```

Each column is now internally uniform — the two ranks in column $x=0$ hold the same pair of chunks, and same for column $x=1$. But different columns still hold different pairs; one more AG phase is needed.

*Phase 4 (X-dim AG, two rings in parallel):* rows run the final AG. Row $y=0$'s ring has $(0,0)$ holding chunks $\{0, 1\}$ and $(1,0)$ holding chunks $\{2, 3\}$ — they exchange their two-chunk payloads. Row $y=1$ same. After this phase every rank holds all four chunks, each of size $M/D_x \cdot D_x = M$.

```
          x=0                                            x=1
       ┌───────────────────────────────────────┬───────────────────────────────────────┐
  y=1  │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁,            │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁,            │
       │  a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]            │  a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]            │
       ├───────────────────────────────────────┼───────────────────────────────────────┤
  y=0  │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁,            │ [a₀+b₀+c₀+d₀, a₁+b₁+c₁+d₁,            │
       │  a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]            │  a₂+b₂+c₂+d₂, a₃+b₃+c₃+d₃]            │
       └───────────────────────────────────────┴───────────────────────────────────────┘
```

AR complete: every rank holds the fully-reduced vector. The AG phases did not introduce any new arithmetic — they just distributed the already-reduced chunks. Total phase count: $2k = 4$ for this 2D example, scaling to $2k = 6$ for a 3D torus.

**Why AG reverses the RS dim order.** The pairing is what makes the cost derivation telescope cleanly. Phase 1 (X-RS) shrinks chunks by $D_x$; the matching Phase 4 (X-AG) grows them back by $D_x$. Phase 2 (Y-RS) shrinks by $D_y$; the matching Phase 3 (Y-AG) grows by $D_y$. Each RS-AG pair moves the same total bytes per rank, and the per-phase BW term depends only on the chunk size at that phase. Running AG in a different dim order would still produce correct output — each dim's AG stage is a self-contained copy — but the chunk-size bookkeeping would not match up against the RS phases and the telescoping derivation below would be messier.

**Why the answer is correct.** Final chunk 0 equals $(a_0 + b_0) + (c_0 + d_0)$ — the two parenthesized terms produced independently in Phase 1, combined in Phase 2. This is valid iff $+$ is associative: $(a + b) + (c + d) = a + b + c + d$. The $k$-dim generalization is the standard factorization of a nested sum:

$$\sum_{x, y, z} V_{x, y, z} \;=\; \sum_z \left( \sum_y \left( \sum_x V_{x, y, z} \right) \right)$$

Each inner sum is an independent problem. Within each dim-phase, all $(N/D_i)$ inner sums for distinct combinations of the *other* coordinates are independent and run in parallel.

**Cost formula and BW telescoping.** Summing α across phases gives $2 \sum_i (D_i - 1)$ sequential hops (RS + AG). Chunk size shrinks geometrically during RS: after phase-1 RS each rank holds $M/D_x$ bytes, after phase 2 $M/(D_x D_y)$, after phase 3 $M/N$ — then grows back symmetrically during AG. Summing the BW term across RS phases telescopes to exactly the flat-ring bound:

$$\frac{M}{\mathrm{BW}} \cdot \left( \frac{D_x - 1}{D_x} + \frac{D_y - 1}{D_x D_y} + \frac{D_z - 1}{D_x D_y D_z} \right) \;=\; \frac{M}{\mathrm{BW}} \cdot \frac{N - 1}{N}$$

AG contributes the same again (each AG phase moves the same bytes as its paired RS phase), so

$$t_{\mathrm{torus,AR}} \;=\; 2 \sum_i (D_i - 1)\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}, \quad N = \prod_i D_i$$

**You pay less α, the same BW as flat ring** — that's the entire arbitrage of dim-decomposition.

**Latency-term compression.** For $N = 512$ at $8 \times 8 \times 8$: $\sum (D_i - 1) = 21$ hops, so $n_\alpha = 42$, versus flat ring's $n_\alpha = 2(N-1) = 1022$ — **24× compression**, from algorithm-topology pairing alone.

**Is dim-decomposition only for a torus?** No. Dim-decomposition is a general technique for any topology whose rank layout factors as a Cartesian product; the torus is canonical because its wiring literally *is* a $k$-D product of rings, so every dim-ring maps onto dedicated physical copper. The main variant within the ring family is **mesh** (no wraparound): each dim-"ring" becomes an open line, so each phase runs a line-based RS / AG (e.g., bidirectional halving) with a slightly higher α cost per dim and a modified BW telescoping — but the dim-decomposition argument itself still holds. Two further cases — hypercube (all $D_i = 2$) and hierarchical fabrics (star + torus / fat-tree composition) — fall more naturally under the tree-based variants in Appendix A and are discussed there.

What the torus specifically buys is the 1:1 mapping of $k$ logical dim-rings onto $k$ physically disjoint link sets, so concurrent dim-phases use literally different copper.

**What the compression requires.**

- **Dim-aligned group layout.** Ranks in the collective group must be contiguous along dim prefixes. A misaligned group (ranks scattered across dim coordinates) falls back to flat-ring bound — see §5 of this note.
- **Wraparound wiring.** Each dim-$i$ ring uses $D_i$ links in a closed loop; see the mesh bullet above for the open-line fallback when wraparound is absent.

**Scaling table.** How dim-decomposed AR's $n_\alpha$ scales with topology choice, fixed $N = 512$:

| Layout | $\sum (D_i - 1)$ | $n_\alpha = 2 \sum(D_i-1)$ | vs. flat ring's $n_\alpha = 1022$ |
|---|---|---|---|
| Flat ring (1D) | 511 | 1022 | 1× |
| 2D torus $32 \times 16$ | 46 | 92 | 11× |
| 3D torus $8 \times 8 \times 8$ | 21 | 42 | 24× |
| 4D torus $8 \times 4 \times 4 \times 4$ | 16 | 32 | 32× |

Diminishing returns beyond 3D: each extra dim saves only a few hops while complicating physical wiring. Real TPU pods use 3D precisely because it's the sweet spot in the compression / wiring trade-off.

**Commercial adoption.** Dim-decomposed ring AR is the default AR primitive on production torus fabrics. Google's TPU v4 / v5p pods (3D torus, scaling up to $16 \times 16 \times 16$ per slice) ship dim-decomposed ring RS+AG as the native cross-replica-sum kernel; XLA / JAX emit it for every sharded reduction, and the TPU v4 paper documents the schedule and reports it running PaLM- and Gemini-class training workloads [TPU-V4]. AWS Trainium2 uses the same primitive on a different substrate: each Trn2 instance is 16 chips wired as a 2D NeuronLink torus (each chip has 4 neighbors), and the Trn2 UltraServer adds a Z-dimension NeuronLink across four instances to form a 3D 64-chip torus [TRN2-ARCH]. AWS's NeuronX Collective Communication Library ships dim-decomposed ring AR (alongside mesh / KangaRing / RDH variants specialized by message size and payload pattern) as its default kernel for these topologies, with dedicated CC-Core hardware orchestrating the per-dim ring phases [NEURON-CC]. Every large-scale model trained on TPU or Trainium relies on this algorithm family for its gradient and activation reductions.

**Floating-point limitation — where it bites in practice.** The limitation below is scoped to **floating-point** reductions only. Integer reductions — quantized-gradient AR, histogram aggregation, counting workloads — are unaffected, because two's-complement integer addition is associative modulo $2^n$: any AR schedule (flat ring, dim-decomposed, tree, hierarchical) produces bit-identical output for integer inputs, with overflow (if any) occurring deterministically across schedules. For floating-point, IEEE-754 addition is non-associative, so reordering the reduction produces numerically-close but not bit-identical outputs — accumulated drift scales as $O(\sqrt{N} \cdot \varepsilon)$ typical over an $N$-way sum, where $\varepsilon$ is the format's relative machine epsilon (FP32 $\approx 1.2 \times 10^{-7}$; BF16 $\approx 7.8 \times 10^{-3}$, ~$10^5\times$ larger per element). For SGD-based training this drift sits far below the optimizer's stochastic noise floor, which is why the algorithm ships by default even under BF16 reductions. It matters only for floating-point workloads that need bit-exact cross-run reproducibility — regression debugging, regulatory ML audits, some scientific workloads — where operators pin the reduction order via fixed algorithm env vars (`NCCL_ALGO=Ring`, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`) and accept the performance cost from forgoing dispatch flexibility.

### 3.2 All-gather / reduce-scatter

AG and RS are the two halves of dim-decomposed AR, so everything from §3.1 carries over directly — the X-phase / Y-phase decomposition, the dim-decomposed ring primitive (with the Rabenseifner-per-dim variant in Appendix A as a reference point), the disjoint-copper parallelism argument, and the chunk-size telescoping. Cost halves because only one half (RS or AG) runs:

$$t_{\mathrm{torus,AG\,or\,RS,ring}} = \sum_i (D_i - 1)\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Commercial shipment.** Dim-decomposed ring AG and RS ship on the same production stacks as the AR variant in §3.1 — XLA / JAX on TPU and NeuronX CCL on Trn2 — because sharded-weight operators (FSDP-style weight gather, ZeRO-3 gradient scatter, sequence-parallel hidden-state gather) need AG and RS as primitives, not just as half of AR. The Rabenseifner-per-dim variant (Appendix A) does not ship as an AG / RS kernel on torus for the same reasons discussed there.

**Additional note on limitations.** The §3.1 limitations (plus the Appendix A caveats if the Rabenseifner path is ever chosen) all apply verbatim with one relaxation: standalone AG has no reduction, so the floating-point non-associativity concern (§3.1's "FP limitation" paragraph) drops out entirely — any AG schedule on any layout produces bit-identical output because bit-copying is exact. Standalone RS still reduces and inherits the AR floating-point behavior unchanged.

### 3.3 All-to-all (bisection-bound)

**Relation to `01_collective_algorithms.md` §5.** The shipped algorithm is pairwise direct-send at $(N-1)\alpha + (N-1)/N \cdot M/\mathrm{BW}$ (§5.1); partner-selection mechanics, chunk-routing logic, and pre/post-rotation bookkeeping port over verbatim. What changes on a torus is the *delivered* bandwidth. A2A has no RS+AG-style decomposition to hide behind (§5 opener makes this explicit: aggregate cross-fabric traffic is $(N-1) M$ bytes regardless of schedule), so unlike AR in §3.1 — where per-dim ring kept torus BW at the star level while only paying more α hops — A2A pays a hard bisection penalty on torus that no scheduling trick can recover on the BW side, while still enjoying the latency compression from dim-decomposition on the α side. **Whether torus helps or hurts vs star therefore depends entirely on which term dominates at the target $(N, M, \alpha, \mathrm{BW})$.**

**Worked example — 4×4 torus.** Take the same 16-rank grid used in §3.1 and cut it in half along the X-axis: bisection passes between column 1 and column 2, and the wraparound bisection passes between column 3 and column 0. Count the severed links:

```
Bisection cut (│ marks severed X-links):

 R_{0,0} ─ R_{1,0} │ R_{2,0} ─ R_{3,0} │ (wrap to R_{0,0})
 R_{0,1} ─ R_{1,1} │ R_{2,1} ─ R_{3,1} │ (wrap to R_{0,1})
 R_{0,2} ─ R_{1,2} │ R_{2,2} ─ R_{3,2} │ (wrap to R_{0,2})
 R_{0,3} ─ R_{1,3} │ R_{2,3} ─ R_{3,3} │ (wrap to R_{0,3})

  2 cuts per row × 4 rows = 8 severed links; each full-duplex at BW.
```

Bisection bandwidth (one direction, summed): $8 \cdot \mathrm{BW}$. Traffic that must cross: every chunk whose source half differs from its destination half. Each rank on the left half ships $N/2 = 8$ chunks (size $M/N = M/16$) to the right half — total left-to-right crossing = $8 \cdot 8 \cdot M/16 = 4 M$. BW term: $4 M / (8\,\mathrm{BW}) = M / (2\,\mathrm{BW})$. The pairwise-direct schedule hits this floor.

**General cost formula.** For a wraparound $k$-D torus with dim sizes $D_i$ and $D_{\max} = \max_i D_i$:

$$\mathrm{BW_{bisect}} = \frac{2 N \cdot \mathrm{BW}}{D_{\max}}, \qquad t_{\mathrm{torus,A2A}} \approx \mathrm{diam} \cdot \alpha + \frac{D_{\max}}{4} \cdot \frac{M}{\mathrm{BW}}, \quad \mathrm{diam} = \sum_i \lfloor D_i / 2 \rfloor$$

The bandwidth term scales with $D_{\max}$ — *not* $N$. On star (the baseline from `01_collective_algorithms.md` §5.1) the equivalent BW term is $(N-1)/N \cdot M/\mathrm{BW} \approx M/\mathrm{BW}$; the torus-vs-star penalty is therefore $D_{\max}/4$. Working a few shapes at $N = 512$:

| Layout | $D_{\max}$ | Torus A2A BW term | Star A2A BW term | Torus penalty |
|---|---|---|---|---|
| Star (fully-connected) | — | — | $\approx M/\mathrm{BW}$ | 1× |
| 3D torus $8 \times 8 \times 8$ | 8 | $2\,M/\mathrm{BW}$ | $\approx M/\mathrm{BW}$ | 2× |
| 3D torus $16 \times 8 \times 4$ | 16 | $4\,M/\mathrm{BW}$ | $\approx M/\mathrm{BW}$ | 4× |
| 3D torus $256 \times 2 \times 2$ (pathological) | 256 | $64\,M/\mathrm{BW}$ | $\approx M/\mathrm{BW}$ | 64× |

Square-ish layouts are strictly better for A2A; long-skinny shapes are catastrophic. This is opposite to AR, where ring-per-dim was insensitive to the shape (total α scaled as $\sum (D_i - 1)$, not $D_{\max}$) — the A2A-vs-AR sensitivity difference is the whole reason torus pods prefer cubic slices for MoE workloads.

**Commercial shipment.** Both pairwise-direct A2A and dim-decomposed A2A (run X-phase row-A2As then Y-phase column-A2As; each phase uses its own dim-copper exactly as in §3.1) ship on TPU and Trainium as the MoE expert-dispatch / expert-combine primitive. The bisection penalty is the dominant design pressure: TPU v4's optical circuit switches [TPU-V4] exist partly so that operators can *reshape* the torus slice into the most cubic layout that matches the MoE's expert count, minimizing $D_{\max}$. Trainium's Trn2 topology [TRN2-ARCH] is fixed at 2D per instance + Z-dim across instances — roughly cubic for 64-chip UltraServers — by the same reasoning. MoE workloads that outgrow a single square slice and must span multiple pods see sharp A2A-cost cliffs, which is why production MoE training/inference pipelines pin expert count to match the slice's natural dimensions whenever possible.

**Limitations (new vs §3.1 / §3.2).**

1. **No decomposition escape for the BW term.** The dim-decomposed AR from §3.1 kept torus BW at star-level by routing each per-dim phase over disjoint copper. A2A's per-dim decomposition still saturates the bisection — splitting into X-phase + Y-phase A2As doesn't add BW, because the total aggregate crossing the cut is fixed by the collective's semantics, not by the schedule. The factor $D_{\max}/4$ penalty is architectural, not algorithmic.
2. **Layout sensitivity is severe.** Unlike AR (§3.1 note), A2A cost scales with $D_{\max}$ rather than $\sum (D_i - 1)$. A $16 \times 8 \times 4$ layout is 2× worse than $8 \times 8 \times 8$ for the same $N = 512$; a $32 \times 4 \times 4$ layout is 4× worse. Operators tuning MoE on torus spend real effort finding the most cubic viable slice.

**What mitigates the bisection penalty.** Two mitigations actually move the A2A cost on torus:

- **Shape choice — minimize $D_{\max}$.** Pick the most cubic layout at the given rank count (cubic > skinny; higher-D > lower-D). For $N = 512$, moving from $32 \times 16$ (2D, $D_{\max} = 32$) to $8 \times 8 \times 8$ (3D, $D_{\max} = 8$) is a 4× BW improvement for free. On TPU v4 / v5p this is a job-launch decision because OCS [TPU-V4] can reconfigure the physical slice shape (and dimensionality) between jobs; on fixed-topology systems like Trainium [TRN2-ARCH] it's baked into the hardware generation.
- **Per-link ICI bandwidth bumps.** Each TPU / Trainium generation raises per-link $\mathrm{BW}$, shrinking the absolute BW term even though the $D_{\max}/4$ coefficient is unchanged.

---

## 4. Side-by-side comparison at N = 512

Everything combined. Assume per-link $\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$. Message size $M = 16\,\mathrm{MB}$ — representative of a TP all-reduce at batched decode or an intermediate-size DP gradient AR in training; an order-of-magnitude anchor, not a workload-specific choice.

### 4.1 All-reduce

| Topology | Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| Star | Ring | 1022 | 511 μs | 35.5 μs | **546 μs** |
| Star | Tree — simple rec. doubling | 9 | 4.5 μs | 160 μs | **165 μs** |
| Star | Tree — Rabenseifner | 18 | 9 μs | 35.5 μs | **45 μs** |
| Star | Tree — DBT (NCCL) | 18 | 9 μs | 35.5 μs | **45 μs** |
| Torus $8 \times 8 \times 8$ | Dim-decomp ring | 42 | 21 μs | 35.5 μs | **57 μs** |

### 4.2 All-gather / reduce-scatter

| Topology | Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| Star | Ring | 511 | 255.5 μs | 17.74 μs | **~273 μs** |
| Star | Recursive halving / doubling | 9 | 4.5 μs | 17.74 μs | **~22.2 μs** |
| Torus $8 \times 8 \times 8$ | Dim-decomp ring | 21 | 10.5 μs | 17.74 μs | **~28.2 μs** |

Same BW term across every row — software AG / RS already hits $\mathrm{BW_{eff}} = \mathrm{BW}$ on a full-duplex fabric (per-rank wall-clock $(N-1)/N \cdot M/\mathrm{BW} \approx M/\mathrm{BW}$), so the entire star-vs-torus and ring-vs-tree gap is on the $\alpha$ side. This is the structural reason AG / RS is the $\alpha$-dominated primitive: half the $2\alpha$ traffic and the same BW floor as AR, so $n_\alpha$ choice is the whole story. The follow-on implication for INC — that an in-network AG / RS lifts the $\alpha$ term only, with no BW ceiling headroom — is treated in [`03_in_network_collectives.md §3.4`](03_in_network_collectives.md#34-cross-primitive-comparison-ag--rs--a2a).

### 4.3 All-to-all

| Topology | Algorithm | $n_\alpha$ | $\alpha$ term | BW term | **Total** |
|---|---|---|---|---|---|
| Star | Pairwise direct-send (NCCL) | 511 | 256 μs | 17.75 μs | **~273 μs** |
| Torus $8 \times 8 \times 8$ | Bisection bound (TPU / Trainium) | 12 | 6 μs | 35.5 μs | **~42 μs** |

Restricted to shipped algorithms, torus wins at these $(N, M)$ by $\sim$6.5×: star pairwise pays $(N-1)\alpha = 511\alpha$ with no latency relief on the shipped menu, while torus's bisection penalty on the BW side ($D_{\max}/4 = 2\times$) is outweighed by its dim-decomposed α compression ($\mathrm{diam} = 12$ vs 511). The gap widens on scale-out fabrics where α is in the μs range rather than the NVLink-class 0.5 μs assumed here.

### 4.4 Key observations

1. **Star+tree AR is the theoretical winner** on small-to-medium $M$ when the switch radix accommodates $N$. Once $N$ exceeds radix, star falls back to a multi-tier fabric and loses its edge.
2. **Torus is competitive at $N = 512$ under ring AR pairing.** It compresses $n_\alpha$ from 1022 → 42 with identical BW term, closing most of the gap to star+tree.
3. **If you swap star's algorithm to ring, star loses** — 546 μs vs torus's 57 μs. This is the subtle point: star's advantage is *algorithmic freedom*, not inherent superiority. The primitive `ring_all_reduce` called on a flat crossbar delivers the flat-ring cost regardless of what the switch could in principle support.
4. **Star and torus are close at $N = 512$ for AR.** Star+DBT's 45 μs vs torus's 57 μs is ~25% — well within the margin that concurrent-group contention and off-prefix layouts can move (see `04_contention_and_congestion.md`). The ideal-model ranking is sensitive to those real-cluster effects.
5. **A2A ranking, shipped-only, is α-dominated on star.** NCCL ships pairwise direct-send at $(N-1)\alpha$ on the latency side and $(N-1)/N \cdot M/\mathrm{BW}$ on the BW side, so star's A2A cost is controlled by the α term once $N$ grows — at $N = 512$, $\alpha = 0.5\,\mu$s, the α term is 256 μs vs BW's 17.75 μs. Torus bisection-bound pairwise pays a $D_{\max}/4$ BW penalty but only $\mathrm{diam}\cdot\alpha$ on the latency side; at symmetric $8 \times 8 \times 8$ this is 35.5 μs BW + 6 μs α = 42 μs total, ~6.5× below star. Asymmetric torus layouts ($D_{\max}$ large) erase this by inflating the BW term — $32 \times 4 \times 4$ pushes BW to 142 μs and turns the ranking back in star's favor. Layout choice is the dominant lever on torus A2A cost; star has no corresponding knob because its per-port bound is layout-independent.

---

## 5. Summary and limitations

### 5.1 Cost summary by topology

Cost table for the primitives above, filtered to algorithms actually shipped by NCCL / RCCL on star and TPU / Trainium on torus (idealized — no contention, uniform per-link BW, per-link $\alpha$; torus rows assume contiguous dim-aligned group placement). Section references cite the derivation in `01_collective_algorithms.md` or earlier in this note.

| Topology | Primitive | Algorithm | Latency term | BW term | Shipped by |
|---|---|---|---|---|---|
| **Star** | AR | Ring | $2(N-1)\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | NCCL / RCCL (small $N$) |
|  | AR | Double binary tree | $2\,\lceil \log_2 N \rceil\,\alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | NCCL / RCCL (multi-node) |
|  | AG / RS | Ring | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | NCCL / RCCL |
|  | A2A | Ring / pairwise | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | NCCL / RCCL |
| **Torus** | AR | Dim-decomp ring | $2 \sum_i (D_i - 1)\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | TPU (XLA / JAX), Trainium (NeuronX CCL) |
|  | AG / RS | Dim-decomp ring | $\sum_i (D_i - 1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | TPU, Trainium |
|  | A2A | Pairwise — bisection-bound | $\mathrm{diam}\cdot\alpha$ | $D_{\max}/4 \cdot M/\mathrm{BW}$ | TPU, Trainium |

Four observations that the rest of this series builds on:

1. **Torus preserves BW-optimality for AR / AG / RS but pays in α.** The torus ring entries keep the same $(N-1)/N$ (or $2(N-1)/N$) BW coefficient as their star counterparts — dim-decomposition routes each phase over disjoint per-dim copper, so no BW is wasted. The torus-vs-star gap is entirely in the α term: $\sum (D_i - 1)$ vs star's $\log_2 N$ (DBT) or $N - 1$ (ring). Production uses ring on both sides because the α gap is small once $M$ is large.
2. **Torus pays a hard BW penalty on A2A.** The $D_{\max}/4 \cdot M/\mathrm{BW}$ BW term has no algorithmic escape — it's set by the bisection cut, not by the schedule. The star-vs-torus penalty scales with $D_{\max}$ (1× at cubic; 64× at $256 \times 2 \times 2$), which is why torus pods aggressively reshape slices toward cubic layouts for MoE workloads.
3. **Tree-flavored algorithms ship on star but not on torus.** DBT is the NCCL / RCCL default on switched fabrics and is selected by the tuner for small-$M$ AR; the structurally analogous dim-decomposed Rabenseifner variant on torus (Appendix A) is absent from both TPU and Trainium runtime stacks because the 1.5–4× α compression at production dim sizes $D_i \in \{4, 8, 16\}$ rarely beats the simpler ring kernel in practice. This is a fabric-economics decision, not a correctness one.
4. **Star has an additional α-compression escape hatch via in-network collectives — torus does not.** Within a switched fabric, the reduction operation can move into the switch ASIC itself: switch-resident ALUs reduce flits on the fly and multicast the result back, collapsing $n_\alpha$ from $\log_2 N$ (DBT) all the way to $2$, independent of $N$. NVLS (NVSwitch), Quantum SHARP (InfiniBand fat-tree / Clos), and Tomahawk Ultra INC (Ethernet) all exploit this. Torus has no analogous path because there is no switch-hosted ALU in the reduce path — the $N$-dependent $\alpha$ term comes from neighbor-router hops, which cannot be collapsed by moving the reduce into a central switch that does not exist. See [`03_in_network_collectives.md`](03_in_network_collectives.md) for the full mechanism and cost model.

### 5.2 Limitations

Every cost formula above assumes the collective runs **alone** with perfect link-level scheduling. Real deployments break this in four ways; `04_contention_and_congestion.md` covers each with coefficient models.

1. **Concurrent collective groups.** DP replicas simultaneously issue TP all-reduces. On a star, non-overlapping port subsets handle them trivially. On torus, if all replicas share a common dim, they contend for the same physical links. Best-case torus assumes this doesn't happen.
2. **Off-prefix group layouts.** Torus dim-decomp only hits $2 \sum(D_i - 1)$ hops when the group ranks are contiguous along dim prefixes. A scatter-pattern allocator (typical in shared clusters) produces groups whose ranks are spread arbitrarily — the physical hop count can be 2–4× the ideal, and the formula falls back to flat-ring bound.
3. **Skewed A2A traffic.** Real MoE routing shows 3–10× skew between hot and cold experts. The uniform-bisection bound underestimates tail latency. Star shrugs this off (every port has the same BW); torus pays — hot-expert destinations concentrated along one dim saturate that dim's bisection well before the uniform bound predicts.
4. **Mixed traffic classes.** Gradient AR, activation AG, optimizer RS, KV cache streaming, and control messages compete for the same fabric — the specifics depend on workload (training vs inference, dense vs MoE), but the structural point is the same: wormhole / cut-through routing keeps the BW per message stable at peak utilization but adds queuing delay at high saturation.

Also worth noting: everything in §4 assumes the software picks the right algorithm for the fabric. Any dispatch layer (NCCL, MPI, a custom tuner) that maps a fabric to its matching algorithm delivers these costs; one that forces flat-ring AR on a torus will see the 511 μs flat-ring cost, not the 21 μs dim-decomp cost — the formula follows the algorithm, not the wiring.

---

## Appendix A: Dim-decomposed Rabenseifner halving-doubling

This appendix preserves the full derivation of the tree-flavored torus AR variant for reference. **It is not shipping** on any production torus stack — neither XLA / JAX on TPU nor NeuronX CCL on Trainium chooses it — and the §3.1 main-text discussion selects dim-decomposed ring instead. The material here exists so the comparison remains complete: it sharpens *why* ring-per-dim wins on torus by showing exactly what the Rabenseifner-per-dim alternative gives up.

**Relation to §3.1.** The compositional framework is identical: run $k$ serial dim-phases for RS, then $k$ more for AG, with $(N/D_i)$ concurrent rings per phase on disjoint copper. The only swap is the inner per-dim primitive — replace each dim's $D_i$-rank ring RS (or AG) with the $D_i$-rank recursive halving-doubling schedule from [`01_collective_algorithms.md`](01_collective_algorithms.md) Appendix A.2 (AR) / Appendix B.1 (standalone RS or AG) (power-of-2 $D_i$ required; for non-power-of-2 $D_i$ the schedule reduces to ring or to a hybrid form). The dim-decomposition argument itself — associativity of reduction, chunk-size telescoping across phases, concurrent ring execution on disjoint copper — is unchanged.

**Why swap the inner primitive?** Recursive halving-doubling takes $\lceil \log_2 D_i \rceil$ α hops per dim-phase instead of $(D_i - 1)$ for ring, while keeping the same $(D_i - 1)/D_i \cdot \mathrm{chunk}/\mathrm{BW}$ bandwidth term per phase (because each rank still sends $(D_i - 1)/D_i$ of its current chunk, just in $\log_2 D_i$ bursts of doubling size rather than $D_i - 1$ bursts of fixed size). The per-dim α compression is modest — $D_i = 4 \to 2$ hops vs 3 for ring (1.5×); $D_i = 8 \to 3$ vs 7 (2.3×); $D_i = 16 \to 4$ vs 15 (3.75×) — but compounds across dims.

**Worked example — 4×4 torus with Rabenseifner per dim.** Same 2D grid as §3.1, but each row/column now runs halving-doubling instead of ring. On $D_x = 4$ the X-phase RS runs $\lceil \log_2 4 \rceil = 2$ butterfly steps per row (step 1: exchange halves with partner 2 away; step 2: exchange quarters with partner 1 away), giving $n_\alpha = 2$ per phase versus ring's $n_\alpha = 3$. The bandwidth telescoping matches ring's exactly — after the X-phase RS each rank holds $M/4$ bytes, after Y-phase RS $M/16$ — because halving-doubling's BW coefficient $(D-1)/D$ is identical to ring's for the same chunk math. Total for 2D AR: $n_\alpha = 2 \cdot 2 \cdot 2 = 8$ versus ring's $2 \cdot 2 \cdot 3 = 12$, with identical BW term.

**Cost formula.** Summing $\lceil \log_2 D_i \rceil$ α hops per dim per half (RS or AG) and applying the same BW telescoping as §3.1:

$$t_{\mathrm{torus,AR,Rab}} \;=\; 2 \sum_i \lceil \log_2 D_i \rceil\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}, \quad N = \prod_i D_i$$

For the AG / RS halves:

$$t_{\mathrm{torus,AG\,or\,RS,Rab}} = \sum_i \lceil \log_2 D_i \rceil\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Latency compression at $N = 512$.** For the $8 \times 8 \times 8$ layout: $\sum \lceil \log_2 D_i \rceil = 9$ hops, so $n_\alpha = 18$ for AR versus dim-decomp ring's 42 (2.3× compression) and flat ring's 1022 (57× compression). The marginal step from ring-per-dim (42) to Rabenseifner-per-dim (18) saves 24 α hops; at $\alpha = 0.5\,\mu$s that is 12 μs — meaningful at very small $M$, negligible once the BW term crosses a few tens of μs.

**Why it does not ship on production torus fabrics.** Four constraints together:

1. **Power-of-2 dim sizes required.** The halving-doubling schedule assumes $D_i$ is a power of 2 along each active dim. TPU v4 / v5p pods configured via OCS can choose $D_i \in \{2, 4, 8, 16\}$ per dim — compatible — but Trainium's fixed 2D-plus-Z 64-chip shape bakes in $D_i = \{4, 4, 4\}$. Any dim mismatch forces fallback to ring-per-dim for that dim, eroding the compression.
2. **Small per-dim $D_i$ limits the α savings.** At $D_i = 4$ the α compression is 1.5× (3 hops → 2); at $D_i = 8$ it is 2.3×. The $\log_2 D_i$ advantage only widens for large $D_i$, but production dim sizes top out around 16 because larger dims degrade A2A (§3.3 bisection-penalty scales as $D_{\max}$).
3. **Kernel complexity dominates the α savings at bulk $M$.** Each halving-doubling step has butterfly-pattern neighbor exchange with doubling chunk size; mapping this onto the torus's fixed $2k$-neighbor wiring requires per-step chunk recomputation on the chip's reduce-engine. The ring kernel uses a single static schedule. For the $M \geq$ few hundred KB bulk regime where production workloads live, the BW term dominates and the extra hops the Rabenseifner variant saves do not pay for the kernel-complexity overhead.
4. **α term is not on the critical path once BW dominates.** At $N = 512$, $\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$, and $M = 16\,\mathrm{MB}$, the ring-per-dim AR cost is 21 μs α + 35.5 μs BW = 57 μs total; Rabenseifner-per-dim trims this to 9 μs α + 35.5 μs BW = 45 μs — a 21% saving on paper that is routinely erased by the kernel-complexity overhead in point 3.

This is a fabric-economics and software-simplicity decision, not a correctness one: dim-decomposed Rabenseifner-per-dim produces the same numerical result (up to floating-point ordering) as dim-decomposed ring, and either could be implemented on the same hardware. Production stacks ship the simpler ring-per-dim kernel because its α disadvantage is small at production dim sizes and its BW coefficient is exactly the same.

**Limitations.** All of §3.1's floating-point-associativity, dim-aligned-group-layout, and wraparound-wiring caveats apply verbatim. The per-dim halving-doubling schedule adds the power-of-2 $D_i$ constraint on top.

---

## Further reading

- **`01_collective_algorithms.md`** — topology-free derivations of the ring, tree, RS, AG, and A2A primitives. Prerequisite for the per-topology cost formulas above.
- **`03_in_network_collectives.md`** — how SHARP / NVLS collapses the $O(N)$ endpoint-hop cost on star topologies to $O(1)$ switch-hop cost.
- **`04_contention_and_congestion.md`** — extending the ideal formulas above with $\eta_\alpha$, $\eta_\beta$ coefficients to score concurrent-group and off-prefix effects.
- **`references.md`** — primary-source citations for the cost formulas in this note (Patarasuk-Yuan bandwidth-optimal AR, Chan et al. dim-decomposed AR).
