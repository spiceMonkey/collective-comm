# Collective Algorithms: A Topology-Free Introduction

**Author:** Yue Lu  
**Date:** April 2026  

Collective operations are the vocabulary of distributed GPU workloads — all-reduce, all-gather, reduce-scatter, all-to-all, point-to-point. They show up in LLM training, LLM inference, HPC simulation, and any other multi-rank compute that needs to synchronize or redistribute state. This note defines each primitive, walks through the two dominant all-reduce algorithms (ring and tree) on small concrete examples, and maps every primitive to the parallelism pattern (TP, EP, SP, PP) that uses it. No physical topology yet — just algorithms as they'd run on an abstract fully-connected fabric. Topology enters in the companion note `02_topology_mapping.md`.

# Table of Contents

1. [The α-β cost model](#1-the-α-β-cost-model)
2. [The seven primitives](#2-the-seven-primitives)
3. [All-reduce](#3-all-reduce)
   - 3.1 [Ring-based AR](#31-ring-based-ar)
   - 3.2 [Tree-based AR — double binary tree](#32-tree-based-ar--double-binary-tree)
   - 3.3 [Comparison and NCCL selection](#33-comparison-and-nccl-selection)
4. [All-gather / reduce-scatter](#4-all-gather--reduce-scatter)
   - 4.1 [Ring-based AG / RS](#41-ring-based-ag--rs)
   - 4.2 [Comparison and NCCL selection](#42-comparison-and-nccl-selection)
5. [All-to-all](#5-all-to-all)
   - 5.1 [Pairwise direct-send A2A](#51-pairwise-direct-send-a2a)
   - 5.2 [Comparison and NCCL selection](#52-comparison-and-nccl-selection)
6. [Point-to-point hop](#6-point-to-point-hop)
7. [Mapping primitives to DP, TP, EP, SP, PP](#7-mapping-primitives-to-dp-tp-ep-sp-pp)
8. [Cost summary table](#8-cost-summary-table)
9. [Appendix A: Non-mainline AR variants](#appendix-a-non-mainline-ar-variants)
   - A.1 [Simple recursive-doubling AR](#a1-simple-recursive-doubling-ar)
   - A.2 [Rabenseifner halving-doubling AR](#a2-rabenseifner-halving-doubling-ar)
10. [Appendix B: Non-mainline AG / RS / A2A variants](#appendix-b-non-mainline-ag--rs--a2a-variants)
    - B.1 [Recursive-doubling AG / recursive-halving RS](#b1-recursive-doubling-ag--recursive-halving-rs)
    - B.2 [Parallel Aggregated Trees (PAT) — scale-out AG / RS](#b2-parallel-aggregated-trees-pat--scale-out-ag--rs)
    - B.3 [Bruck A2A](#b3-bruck-a2a)
11. [Further reading](#further-reading)

---

## 1. The α-β cost model

Every collective algorithm on every topology costs time in the same two-part shape. The time to move a message of $M$ bytes between two ranks on a link of bandwidth $\mathrm{BW}$ is modeled as

$$t = \alpha + \frac{M}{\mathrm{BW}}$$

- $\alpha$ (**per-hop latency**): fixed setup + switch traversal + propagation, measured in seconds. Does not depend on message size.
- $M / \mathrm{BW}$ (**bandwidth term**): time spent pushing bytes through the wire.

A collective that takes $k$ sequential hops moving per-hop payload $m$ costs

$$t = k \cdot \alpha + k \cdot \frac{m}{\mathrm{BW}}$$

Two knobs differentiate algorithms:

- **Latency term** — how many sequential hops does the schedule need? A ring of $N$ ranks chains $N-1$ sequential steps; a tree chains $\lceil \log_2 N \rceil$.
- **Bandwidth term** — how much data does each rank push, summed across the schedule? Good algorithms keep this close to the lower bound $M \cdot (N-1)/N$ (you can't avoid shipping one rank's share across the cut).

Throughout this note, "cost" means wall-clock time under this model. Contention and congestion (which inflate $\alpha$ and deflate $\mathrm{BW}$) are covered in `04_contention_and_congestion.md`.

---

## 2. The seven primitives

Seven collectives cover essentially all multi-rank communication in distributed GPU workloads (LLM training and inference, HPC simulation, distributed gradient descent, etc.). The first six are defined on a group of $N$ ranks each holding data of size $M$; the last is point-to-point. In the table below, $V$ denotes a payload of size $M$ bytes (treated as a vector for reduction purposes — e.g., a gradient tensor, an activation shard, a KV chunk); $V_i$ is the copy held by rank $i$, and $\sum_i V_i$ is the element-wise sum across ranks. For AR / RS the reduction operator is commutative-associative (sum, max, min, bit-op); broadcast / AG / A2A just move data and don't combine it.

| Primitive | What each rank starts with | What each rank ends with |
|---|---|---|
| **Broadcast** | rank 0 holds $V$; others hold nothing | all ranks hold $V$ |
| **Reduce** | rank $i$ holds $V_i$ | rank 0 holds $\sum_i V_i$ (others unchanged) |
| **All-reduce (AR)** | rank $i$ holds $V_i$ | all ranks hold $\sum_i V_i$ |
| **Reduce-scatter (RS)** | rank $i$ holds length-$M$ vector $V_i$ | rank $i$ holds one chunk of size $M/N$ equal to $\sum_j V_j[\mathrm{chunk}_i]$ |
| **All-gather (AG)** | rank $i$ holds one chunk of size $M/N$ | all ranks hold the concatenation of all $N$ chunks |
| **All-to-all (A2A)** | rank $i$ holds $N$ chunks of size $M/N$; chunk $j$ is destined for rank $j$ | rank $i$ holds $N$ chunks of size $M/N$; chunk $j$ was sent by rank $j$ (specifically, rank $j$'s original "chunk $i$") |
| **P2P (send/recv)** | rank $a$ holds $V$ | rank $b$ holds $V$ |

*A2A indexing convention:* chunks in the **send** buffer are indexed by **destination** (rank $i$'s chunk $j$ is what rank $i$ wants to give to rank $j$); chunks in the **receive** buffer are indexed by **sender** (rank $i$'s received chunk $j$ is what rank $j$ gave it, which was rank $j$'s original chunk $i$). This matches the MPI `MPI_Alltoall` layout.

Two useful identities:

- **AR ≡ RS + AG** (*logically*). The end state of AR is identical to running RS then AG: RS leaves each rank with one fully-reduced $M/N$-chunk, and AG then redistributes those chunks so everyone has all $N$ of them. This is a semantic equivalence, not a prescription — some AR algorithms (bandwidth-optimal **ring** AR [Patarasuk-Yuan 2009], dim-decomposed ring AR on torus/mesh) do execute as a literal RS phase followed by an AG phase, but others (**recursive halving-doubling** / Rabenseifner, **recursive doubling**, tree AR, SHARP-style in-network AR) never materialize the intermediate RS end-state as a distinct phase. The identity is still useful for reasoning about cost lower bounds (§3) regardless of implementation.
- **A2A is a permutation, not a reduction**. No summation happens; every rank ends up holding data that belongs elsewhere. MoE expert dispatch is exactly this pattern (plus a reverse A2A for the combine).

Sections 3–5 walk through AR, AG/RS, and A2A — the three reduction / redistribution primitives that dominate time in any collective-heavy workload (DDP/FSDP training, LLM inference, MoE routing, HPC all-reduce). Each section follows the same template: a ring-based derivation (the NCCL large-$M$ default for all three primitives), the tree-based variant that NCCL actually ships alongside ring when one exists (double binary tree for AR), and a comparison that explains the NCCL selection rule between them. Variants that appear in the literature and in other runtimes — simple recursive-doubling AR, Rabenseifner halving-doubling AR, recursive-doubling AG / recursive-halving RS, Parallel Aggregated Trees (PAT), Bruck A2A — are not in the NCCL shipping menu but remain useful reference points; their full step-by-step derivations live in Appendix A (AR variants) and Appendix B (AG / RS / A2A variants), cross-referenced inline where each comparison in the main sections calls for them. §6 covers point-to-point, §7 maps primitives to parallelism axes, and §8 is the cost summary.

---

## 3. All-reduce

AR is the workhorse reduction primitive: every rank starts with its own length-$M$ vector $V_i$ and ends holding the elementwise sum $\sum_i V_i$. Two families of algorithms dominate in practice: **ring-based** (§3.1), which chains $2(N-1)$ sequential exchanges along a logical ring and hits the Patarasuk-Yuan bandwidth lower bound, and **tree-based** (§3.2), which collapses the step count to $O(\log N)$ via hypercube / binary-tree pairings. NCCL ships ring and double binary tree (DBT) — the two algorithms covered in the main text below. Two tree variants from the MPI literature and older HPC work — simple recursive doubling and Rabenseifner halving-doubling — remain instructive as reference points even though they are not in NCCL's shipping menu; their derivations live in Appendix A so the comparison in §3.3 can cite concrete cost formulas. §3.3 summarizes the four-way comparison and explains the NCCL selection rule.

### 3.1 Ring-based AR

Ring AR on $N$ ranks runs RS in $N-1$ steps, then AG in $N-1$ steps — the Patarasuk-Yuan bandwidth-optimal construction. We walk through $N=4$, each rank holding a length-4 vector, chunked into 4 pieces.

**Setup.** Four ranks $R_0, R_1, R_2, R_3$ in a ring $R_0 \to R_1 \to R_2 \to R_3 \to R_0$. Each rank $R_i$ holds a vector $V_i = [v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}]$. Target: every rank ends up holding $[S_0, S_1, S_2, S_3]$ where $S_k = \sum_i v_{i,k}$.

**Phase 1: reduce-scatter (3 steps).**

At step $t \in \{1, 2, \ldots, N-1\}$, rank $i$ **sends** chunk $(i - t + 1) \bmod N$ to its right neighbor and **receives** chunk $(i - t) \bmod N$ from its left neighbor, adding the incoming value into its local copy at that slot. That slot — $(i - t) \bmod N$ — is the rank's **accumulator chunk** for step $t$. Sanity-check at $t=1$: $R_i$ sends its own-index chunk $i$ rightward and accumulates into slot $(i-1) \bmod N$.

The accumulator slot index shifts by $-1 \bmod N$ every step. After $N-1$ steps, each rank's accumulator sits at slot $(i - (N-1)) \bmod N = (i+1) \bmod N$ and holds the fully reduced $N$-way sum for that slot.

```
Initial (before any step):

R0: [v00 v01 v02 v03]
R1: [v10 v11 v12 v13]
R2: [v20 v21 v22 v23]
R3: [v30 v31 v32 v33]
```

**After step 1** — each rank sends one chunk right and receives one chunk from its left neighbor, adding it into its local copy. Only the accumulator chunk changes; every other slot is untouched.

```
Step 1: each rank adds one incoming chunk into its accumulator slot

R0: [v00           v01           v02           v03+v33     ]   ← accumulator: chunk 3
R1: [v10+v00       v11           v12           v13         ]   ← accumulator: chunk 0
R2: [v20           v21+v11       v22           v23         ]   ← accumulator: chunk 1
R3: [v30           v31           v32+v22       v33         ]   ← accumulator: chunk 2
```

Every accumulator now holds a **2-term partial sum**. The slots not marked as accumulators still hold the rank's original values — they'll stay that way until AG phase overwrites them.

**After step 2** — each rank forwards the chunk it just updated to its right neighbor, so the partial sum grows by one more term. The accumulator slot index shifts by $-1 \bmod 4$ each step.

```
Step 2: each rank forwards its step-1 accumulator; receiver adds it in

R0: [v00           v01           v02+v22+v32   v03+v33     ]   ← accumulator: chunk 2 (3 terms)
R1: [v10+v00       v11           v12           v13+v03+v33 ]   ← accumulator: chunk 3 (3 terms)
R2: [v20+v10+v00   v21+v11       v22           v23         ]   ← accumulator: chunk 0 (3 terms)
R3: [v30           v31+v21+v11   v32+v22       v33         ]   ← accumulator: chunk 1 (3 terms)
```

Trace of what flowed into each accumulator:

- R0 chunk 2 picked up $v_{22}+v_{32}$ from R3 (R3's chunk 2 after step 1).
- R1 chunk 3 picked up $v_{03}+v_{33}$ from R0.
- R2 chunk 0 picked up $v_{10}+v_{00}$ from R1.
- R3 chunk 1 picked up $v_{21}+v_{11}$ from R2.

**After step 3** — one more forward, and each accumulator absorbs the final term. The accumulator is now a **full 4-way sum** — that's $S_k$, the fully-reduced chunk for slot $k$.

```
Step 3: each rank forwards its step-2 accumulator; receiver adds the last summand

R0: [v00           S1            v02+v22+v32   v03+v33     ]   ← fully reduced: chunk 1 = S1
R1: [v10+v00       v11           S2            v13+v03+v33 ]   ← fully reduced: chunk 2 = S2
R2: [v20+v10+v00   v21+v11       v22           S3          ]   ← fully reduced: chunk 3 = S3
R3: [S0            v31+v21+v11   v32+v22       v33         ]   ← fully reduced: chunk 0 = S0

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

This is exactly the RS end-state: each rank holds **one** fully-reduced chunk (on a different slot), and the other three slots still contain junk partial sums. Those will be overwritten in the next phase.

**Cost accounting.** Each of the 3 steps costs $\alpha + M/(4\,\mathrm{BW})$ (one handshake plus one $M/N$-sized chunk over the link). The 3 steps run **sequentially** (step $t+1$ forwards the chunk that step $t$ just accumulated), so the costs add: the whole RS phase takes

$$t_{\mathrm{RS}} = 3\alpha + 3 \cdot \frac{M}{4\,\mathrm{BW}} \qquad \text{(totals across all 3 steps, not per step).}$$

Generalizing to $N$ ranks: $(N-1)\alpha + (N-1)\,M/(N\,\mathrm{BW})$.

**Phase 2: all-gather (3 steps).**

After RS, each rank owns exactly one fully-reduced chunk — $R_i$ owns slot $(i+1) \bmod N$ — and needs the other three. The plan: same ring, same direction (send right, receive from left), **no reduction** — each rank forwards what it just received, overwriting the junk in the receiver's corresponding slot.

At step $t \in \{1, 2, \ldots, N-1\}$, rank $i$ **sends** chunk $(i - t + 2) \bmod N$ to its right neighbor and **overwrites** its local slot $(i - t + 1) \bmod N$ with the chunk received from its left neighbor. Sanity-check at $t=1$: $R_i$ forwards its freshly-reduced slot $(i+1) \bmod N$ rightward and overwrites its own-index slot $i$ with an incoming fully-reduced chunk.

These are exactly the RS formulas **shifted by $+1$ in slot index**. Same ring, same direction, same $-1 \bmod N$ slot-index shift per step — just a $+1$ offset in where the action starts. The reason for the offset: RS leaves $R_i$ owning slot $(i+1) \bmod N$ (not slot $i$), so AG begins by forwarding that $+1$-offset slot.

Starting state (using `?` for the stale partial sums from RS — they're unreadable garbage, about to be overwritten):

```
Before AG step 1 (only the fully-reduced chunk per rank is known-good):

R0: [ ?     S1    ?     ?  ]
R1: [ ?     ?    S2     ?  ]
R2: [ ?     ?     ?    S3  ]
R3: [S0     ?     ?     ?  ]
```

**After AG step 1** — each rank sends its one fully-reduced chunk right. The receiver overwrites its corresponding slot.

```
AG step 1: R_i sends its one known chunk to R_{i+1}

R0: [S0    S1    ?     ?  ]   (received S0 from R3 → chunk 0 overwritten)
R1: [ ?    S1   S2     ?  ]   (received S1 from R0 → chunk 1 overwritten)
R2: [ ?    ?    S2    S3  ]   (received S2 from R1 → chunk 2 overwritten)
R3: [S0    ?     ?    S3  ]   (received S3 from R2 → chunk 3 overwritten)
```

Each rank now holds **two** fully-reduced chunks.

**After AG step 2** — each rank forwards the chunk it *just* received (not its original one) to the right. Same overwrite pattern.

```
AG step 2: forward what you just received

R0: [S0    S1    ?    S3  ]   (received S3 from R3 → chunk 3 overwritten)
R1: [S0    S1   S2     ?  ]   (received S0 from R0 → chunk 0 overwritten)
R2: [ ?    S1   S2    S3  ]   (received S1 from R1 → chunk 1 overwritten)
R3: [S0    ?    S2    S3  ]   (received S2 from R2 → chunk 2 overwritten)
```

Three of the four slots are now full-reduced on every rank.

**After AG step 3** — one last forward completes the ring.

```
AG step 3: forward what you just received

R0: [S0    S1   S2    S3  ]
R1: [S0    S1   S2    S3  ]
R2: [S0    S1   S2    S3  ]
R3: [S0    S1   S2    S3  ]
```

Every rank now holds the complete reduced vector $[S_0, S_1, S_2, S_3]$. AR is done.

**Cost accounting (AG).** Same structure as RS — 3 sequential steps, each $\alpha + M/(4\,\mathrm{BW})$. Total for the AG phase:

$$t_{\mathrm{AG}} = 3\alpha + 3 \cdot \frac{M}{4\,\mathrm{BW}} \qquad \text{(totals across all 3 steps).}$$

Generalizing to $N$ ranks: $(N-1)\alpha + (N-1)\,M/(N\,\mathrm{BW})$.

**Total cost.** Combining RS + AG for $N$ ranks:

$$t_{\mathrm{ring\,AR}} = 2(N-1)\,\alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

For $N=4$: $6\alpha + 1.5 \cdot M/\mathrm{BW}$. The bandwidth term saturates at $2 M/\mathrm{BW}$ as $N \to \infty$ — every byte needs to leave the originating rank once and arrive at the other $N-1$ ranks once, and the ring pipelines this nearly perfectly.

**Strengths.** Bandwidth-optimal. Simple. Works on any fabric that can route a logical ring (including torus when wraparound is used).

**Weakness.** Latency term grows **linearly** with $N$. At $N=512$, that's $1022\,\alpha$ — a penalty heavy enough to motivate the tree-based algorithms covered in §3.2, which replace the linear $N$ scaling with $\log_2 N$ at the cost of a worse bandwidth constant.

### 3.2 Tree-based AR — double binary tree

Tree-based AR collapses ring's $N-1$ sequential steps into $\lceil \log_2 N \rceil$ by using a **binary-hypercube pairing pattern** instead of a ring. At each step, every rank exchanges data with exactly one partner, and the "reachable set" doubles in size each step — so the whole $N$-rank group is covered in $\log_2 N$ steps.

Three tree-flavored AR variants appear in the literature, each reading AR through one of its two natural decompositions (RS + AG, or reduce + broadcast):

- **Simple recursive doubling — does not decompose.** Single-pass butterfly / hypercube sweep, equivalent to recursive-doubling AG with addition substituted for concatenation (adding two $M$-sized vectors keeps the result $M$-sized, so no half grows or shrinks). Minimum latency, but the price of single-pass simplicity is sending the full $M$ at every step, so bandwidth scales as $\log_2 N \cdot M$. **Not shipped by NCCL.** Full derivation: [Appendix A.1](#a1-simple-recursive-doubling-ar).
- **Rabenseifner halving-doubling ≡ RS + AG.** Two complementary hypercube passes with chunk-exponential payloads. Recognizes that step $k$ only needs the step-$k$-relevant chunks (size $2^{k-1} \cdot M/N$ or its complement), which shrinks the BW coefficient from $\log_2 N$ to the optimal $2(N{-}1)/N$ at the cost of $2\times$ more steps. **Not shipped by NCCL.** Full derivation: [Appendix A.2](#a2-rabenseifner-halving-doubling-ar).
- **Double binary tree (DBT) ≡ reduce + broadcast (× 2 trees).** Two complementary tree passes, full-vector payloads split into pipeline segments. The second tree $T_2$ runs concurrently with complementary interior/leaf roles so both link directions stay busy, halving the per-link message at every step. **Shipped by NCCL** as the non-ring multi-node default; derived in full below.

**Why NCCL ships DBT and not the other two is the subject of §3.3**; the short version is that DBT is the only tree-flavored variant whose per-rank concurrency requirement fits the port budget of real fabrics, which in turn lets pipelining collapse its $\log_2 N$ BW coefficient down to near-ring. The other two serialize on bounded-port fabrics.

**DBT construction** [SST09] reaches $O(\log N)$ latency with near-optimal bandwidth by running a single binary tree AR (reduce up + broadcast down, $2 \log_2 N$ steps), then "doubling" it with a second tree of complementary roles concurrently driving both directions of each link. We build it up in two parts: first trace the single-tree AR at $N=4$, then layer on the double-tree optimization.

**Single binary tree at $N=4$.** Tree structure:

```
         R0 (root, depth 0)
        /  \
       R1   R2  (depth 1)
      /
     R3  (depth 2)
```

$R_0$ is root. $R_1$ and $R_2$ are $R_0$'s children. $R_3$ is $R_1$'s child. Depth = 2 = $\lceil \log_2 4 \rceil$.

Initial state (same $v_{i,k}$ as before):

```
R0: [v00 v01 v02 v03]
R1: [v10 v11 v12 v13]
R2: [v20 v21 v22 v23]
R3: [v30 v31 v32 v33]
```

**Phase 1 — Reduce (depth $\log_2 N$ steps, leaves → root).**

**Step 1 (depth 2 → depth 1).** $R_3$ sends its full vector up to $R_1$; $R_1$ sums. $R_2$ is idle (no children).

```
R0: [v00         v01         v02         v03       ]   ← unchanged (root waiting)
R1: [v10+v30     v11+v31     v12+v32     v13+v33   ]   ← 2-way sum over {R1, R3}
R2: [v20         v21         v22         v23       ]   ← unchanged (leaf, waiting)
R3: stale                                                ← sent up
```

**Step 2 (depth 1 → depth 0).** $R_1$ and $R_2$ concurrently send their vectors up to $R_0$. $R_0$ sums both incoming into its local copy.

```
R0: [S0          S1          S2          S3        ]   ← 4-way full sum
R1: stale                                                ← sent up
R2: stale                                                ← sent up
R3: stale
```

**Phase 2 — Broadcast (depth $\log_2 N$ steps, root → leaves).**

**Step 3 (depth 0 → depth 1).** $R_0$ concurrently sends $S$ down to $R_1$ and $R_2$.

```
R0: [S0 S1 S2 S3]
R1: [S0 S1 S2 S3]   ← received
R2: [S0 S1 S2 S3]   ← received
R3: stale
```

**Step 4 (depth 1 → depth 2).** $R_1$ sends $S$ down to $R_3$. $R_2$ is idle (no children).

```
R0: [S0 S1 S2 S3]
R1: [S0 S1 S2 S3]
R2: [S0 S1 S2 S3]
R3: [S0 S1 S2 S3]   ← received
```

All 4 ranks now hold $S$. Single-tree AR done in $2 \lceil \log_2 N \rceil = 4$ sequential steps.

**Cost of single binary tree.** Each step moves the full $M$-byte vector on the active link: $\alpha + M/\mathrm{BW}$. 4 sequential steps → total $4\alpha + 4M/\mathrm{BW}$. Generalizing:

$$t_{\mathrm{single\,tree\,AR}} = 2\lceil \log_2 N \rceil \, \alpha + 2\lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

Same latency as Rabenseifner, but bandwidth is $\log_2 N$ times worse. The inefficiency: at steps 1 and 4, only one link is active; at steps 2 and 3, multiple links are active but each one is driving only one direction. Half the full-duplex link capacity is idle across the algorithm.

**The double-tree optimization.** Construct a second tree $T_2$ whose role assignments are the complement of $T_1$: every rank that is interior in $T_1$ is a leaf in $T_2$, and vice versa. For $N=4$:

```
T1:         R0                T2:          R3
           /  \                            /  \
          R1   R2                         R0   R2
         /                                      \
        R3                                       R1
```

Role check:
| Rank | $T_1$ | $T_2$ |
|---|---|---|
| $R_0$ | root (internal) | leaf |
| $R_1$ | internal | leaf |
| $R_2$ | leaf | internal |
| $R_3$ | leaf | root (internal) |

Each rank is interior in *exactly one* tree. Now run AR on $T_1$ and $T_2$ **concurrently**, each on a different half of the message: chunks $\{0, 1\}$ (left half, $M/2$ bytes) flow through $T_1$ — reduced at $R_0$, broadcast from $R_0$. Chunks $\{2, 3\}$ (right half, $M/2$ bytes) flow through $T_2$ — reduced at $R_3$, broadcast from $R_3$. Because the two trees have complementary roles, a rank's $T_1$ traffic travels on one direction of the full-duplex link while its $T_2$ traffic travels on the other — both halves make progress every step.

Initial state (`|` separates the left half carried by $T_1$ from the right half carried by $T_2$):

```
               left half (T1)        |       right half (T2)
R0: [        v00         v01         |       v02         v03        ]
R1: [        v10         v11         |       v12         v13        ]
R2: [        v20         v21         |       v22         v23        ]
R3: [        v30         v31         |       v32         v33        ]
```

**Step 1 — reduce, depth 2 → 1 in both trees.** $T_1$: $R_3 \to R_1$ (left half). $T_2$: $R_1 \to R_2$ (right half). Concurrent.

```
R0: [        v00         v01         |       v02         v03        ]   ← idle both trees
R1: [       v10+v30    v11+v31       |      stale       stale       ]   T1: +R3 left; T2: right sent up
R2: [        v20         v21         |     v22+v12    v23+v13       ]   T2: +R1 right
R3: [       stale       stale        |       v32         v33        ]   T1: left sent up
```

**Step 2 — reduce, depth 1 → 0 in both trees.** $T_1$: $R_1, R_2 \to R_0$ (left halves, two concurrent sends into $R_0$). $T_2$: $R_0, R_2 \to R_3$ (right halves, two concurrent sends into $R_3$). Concurrent.

```
R0: [         S0          S1         |      stale       stale       ]   T1: 4-way left; T2: right sent up
R1: [       stale       stale        |      stale       stale       ]
R2: [       stale       stale        |      stale       stale       ]
R3: [       stale       stale        |        S2          S3        ]   T2: 4-way right

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

After the two reduce steps, $R_0$ holds the fully-reduced left half and $R_3$ holds the fully-reduced right half — mirror-image completions of each tree's reduce phase.

**Step 3 — broadcast, depth 0 → 1 in both trees.** $T_1$: $R_0 \to R_1, R_2$ (left half). $T_2$: $R_3 \to R_0, R_2$ (right half). Concurrent.

```
R0: [         S0          S1         |        S2          S3        ]   T2: received right from R3
R1: [         S0          S1         |      stale       stale       ]   T1: received left from R0
R2: [         S0          S1         |        S2          S3        ]   both trees: received
R3: [       stale       stale        |        S2          S3        ]
```

**Step 4 — broadcast, depth 1 → 2 in both trees.** $T_1$: $R_1 \to R_3$ (left half). $T_2$: $R_2 \to R_1$ (right half). Concurrent.

```
R0: [         S0          S1         |        S2          S3        ]
R1: [         S0          S1         |        S2          S3        ]   T2: received right from R2
R2: [         S0          S1         |        S2          S3        ]
R3: [         S0          S1         |        S2          S3        ]   T1: received left from R1
```

All 4 ranks now hold $[S_0, S_1, S_2, S_3]$ in $2\lceil \log_2 N \rceil = 4$ sequential steps — same step count as single-tree, but each step moves only $M/2$ bytes per link (the other half is on the sibling tree, which occupies the opposite direction of the same full-duplex link).

**Cost.** $2\lceil \log_2 N \rceil$ sequential steps, each costing $\alpha + (M/2)/\mathrm{BW}$ (handshake + half-message on the active link):

$$t_{\mathrm{double\,tree\,AR}} = 2\lceil \log_2 N \rceil \, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

For $N=4$: $4\alpha + 2\,M/\mathrm{BW}$ — $2\times$ better than single-tree's $4\,M/\mathrm{BW}$, and at this size coincidentally matching ring's $2\,M/\mathrm{BW}$. The bandwidth term improves on single-tree's $2\log_2 N \cdot M/\mathrm{BW}$ by a factor of $2$, thanks to the half-message-per-link property the second tree enables.

**Strengths.** $O(\log N)$ latency (best class). Near-optimal bandwidth via the second tree. Works on non-power-of-2 $N$ with slight asymmetry in the tree construction (unlike Rabenseifner, which needs clean bit-pairings). Adapts well to fat-tree / spine-leaf physical topologies where link capacity is plentiful.

**Weakness.** Two-tree construction and the pipelined segmentation make the runtime considerably more complex than ring or Rabenseifner. The bandwidth advantage depends on full-duplex links — degrades toward single-tree cost on half-duplex or heavily oversubscribed fabrics.

### 3.3 Comparison and practical implementation choices

Side-by-side summary of the four algorithms in §§3.1–3.2, their pure-algorithm costs, and which production libraries ship them. The Latency and Bandwidth columns correspond to two regimes of operation: at small $M$ the collective is **latency-bound** and the $\alpha$ column dominates (algorithms with fewer hops win), at large $M$ it is **bandwidth-bound** and the $M/\mathrm{BW}$ column dominates (algorithms with the smaller BW coefficient win); the crossover sits at $M^{*} \sim \alpha \cdot \mathrm{BW}$, which is a property of the hardware, not of the algorithm. The "Used by" column cites libraries but does **not** imply the library's runtime cost matches the pure formula — real runtimes layer pipelining, topology awareness, and in-network offload on top, which can shift the bandwidth term in particular.

| Algorithm | Latency | Bandwidth | NCCL adoption |
|---|---|---|---|
| Ring (§3.1) | $2(N{-}1)\,\alpha$ | $\dfrac{2(N{-}1)}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Y |
| Simple rec. doubling ([App. A.1](#a1-simple-recursive-doubling-ar)) | $\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot \dfrac{M}{\mathrm{BW}}$ | N |
| Rabenseifner ([App. A.2](#a2-rabenseifner-halving-doubling-ar)) | $2\lceil \log_2 N \rceil \, \alpha$ | $\dfrac{2(N{-}1)}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | N |
| Double binary tree (§3.2) | $2\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot \dfrac{M}{\mathrm{BW}}$ | Y |

**Why NCCL/RCCL ship ring + double binary tree and not Rabenseifner or plain recursive doubling.** The "best on paper" entries in the table — Rabenseifner's bandwidth-optimal BW term, plain recursive doubling's minimum $\log_2 N$ latency — aren't what production libraries actually ship. The dominant reason is **pipelining**, and specifically how each algorithm's structure interacts with the finite number of concurrent communication ports on a real node. We derive the pipelining cost first, then apply it to all four algorithms.

**Why pipeline at all?** Start with the non-pipelined schedule: $L$ sequential steps, each shipping the full $M$ bytes. Total cost $L\alpha + LM/\mathrm{BW}$. The ugly part is the second term — the bandwidth coefficient scales linearly with $L$. The inefficiency is easy to see: while step $k$ is busy pushing bytes, steps $1, \ldots, k{-}1$ sit idle (they finished their part of the transfer several slots ago, but there's nothing queued for them to do). If we could somehow keep every step busy at once, the whole algorithm would finish in roughly the time of *one* step's work regardless of $L$. Pipelining is the trick that makes that happen.

**The conveyor idea.** Cut the message $M$ into $P$ equal sub-segments of size $M/P$ and feed them through the schedule one after another, like parts on an assembly line. Segment 1 enters step 1 at time slot 1, moves to step 2 at slot 2, then to step 3 at slot 3, and so on. Segment 2 enters step 1 at slot 2 (one slot behind segment 1) and trails it through. By the time segment $P$ has entered the pipeline, segments $1, \ldots, P{-}1$ are already distributed across the later steps — and each slot now advances *every* segment by one step simultaneously.

**Space-time diagram.** Rows = steps in the schedule, columns = time slots. Each slot costs exactly $\alpha + (M/P)/\mathrm{BW}$ (one handshake plus one sub-segment's worth of transfer). Cell $(k, t)$ = "segment at step $k$ during slot $t$". Small example with $L=3$ steps and $P=3$ segments:

```
              slot 1    slot 2    slot 3    slot 4    slot 5
    step 1:   seg1      seg2      seg3      ·         ·
    step 2:   ·         seg1      seg2      seg3      ·
    step 3:   ·         ·         seg1      seg2      seg3

              └──── fill ────┘    steady    └──── drain ────┘
                 (P-1 = 2)       (1 slot)        (L-1 = 2)
```

Three phases:
- **Fill** (slots 1 to $P{-}1$) — the pipeline is warming up. Step 1 has work from slot 1; step 2 doesn't start until slot 2; step $L$ doesn't start until slot $L$.
- **Steady state** (slots where all $L$ steps are busy simultaneously). In the picture: slot 3 has $\text{seg3}$ at step 1, $\text{seg2}$ at step 2, $\text{seg1}$ at step 3 — every step advancing one segment per slot. This is the regime where the pipeline pays for itself.
- **Drain** (slots $P+1$ to $L+P{-}1$) — segments that entered late are still finishing. Step 1 has nothing left to do.

**Total slot count: $L + P - 1$.** The assembly-line identity: the *last* segment (segment $P$) enters step 1 at slot $P$ and needs $L - 1$ additional slots to advance through the remaining steps, exiting at slot $P + (L-1) = L + P - 1$. In the diagram: $3 + 3 - 1 = 5$ slots. Matches.

**The formula.** Each slot costs $\alpha + (M/P)/\mathrm{BW}$; there are $L + P - 1$ of them:

$$t_{\mathrm{pipe}}(P) = (L + P - 1)\left(\alpha + \frac{M/P}{\mathrm{BW}}\right)$$

To see *where the L factor goes*, expand the bandwidth piece. Using $(L{+}P{-}1)/P = 1 + (L{-}1)/P$:

$$t_{\mathrm{pipe}}(P) \;=\; \underbrace{(L + P - 1)\,\alpha}_{\substack{\text{hop count}\\ L\alpha\,+\,(P-1)\alpha\\ \text{extra handshakes for fill/drain}}} \;+\; \underbrace{\frac{M}{\mathrm{BW}}}_{\substack{\text{steady-state BW}\\ \text{(saturates at 1}\cdot M/\mathrm{BW}\text{)}}} \;+\; \underbrace{\frac{(L - 1)\,M}{P\,\mathrm{BW}}}_{\substack{\text{fill+drain overhead}\\ \text{(shrinks like }1/P\text{)}}}$$

Three things to notice:
1. **The $L$-dependence has *left* the dominant BW term.** Non-pipelined BW was $L \cdot M/\mathrm{BW}$; here the $L$ survives only inside the fill/drain piece, which vanishes as $P \to \infty$.
2. **The saturated BW floor is $M/\mathrm{BW}$** — the volume a single rank-port must push regardless of $P$ or $L$. You can't go below that without using more ports.
3. **$P$ can't grow arbitrarily large**, because the hop-count term picks up $(P{-}1)\alpha$ extra handshakes. More segments = more slot boundaries = more $\alpha$.

**Sanity check with numbers.** Set $L = 3$.
- **Non-pipelined** ($P = 1$): cost $= 3\alpha + 3M/\mathrm{BW}$. BW coefficient 3.
- **Pipelined with $P = 3$**: $5\alpha + M/\mathrm{BW} + (2/3)M/\mathrm{BW} = 5\alpha + 1.67\,M/\mathrm{BW}$. BW coefficient dropped from 3 to 1.67 at the cost of 2 extra handshakes.
- **Pipelined with $P = 10$**: $12\alpha + M/\mathrm{BW} + 0.2\,M/\mathrm{BW} = 12\alpha + 1.2\,M/\mathrm{BW}$. Within 20 % of the asymptotic floor, at the cost of 9 extra handshakes.
- **$P \to \infty$**: cost $\to \infty \cdot \alpha + M/\mathrm{BW}$. Asymptotic BW is achieved, but latency has exploded — clearly past the optimum.

**Optimal segmentation.** Minimize by differentiating with respect to $P$ (continuous approximation):

$$\frac{dt_{\mathrm{pipe}}}{dP} \;=\; \alpha \;-\; \frac{(L - 1)\,M}{P^2\,\mathrm{BW}} \;=\; 0 \quad\Longrightarrow\quad P^{*} \;=\; \sqrt{\frac{(L - 1)\,M}{\alpha\,\mathrm{BW}}}$$

Substituting back:

$$t_{\mathrm{pipe}}(P^{*}) \;=\; L\,\alpha \;+\; \frac{M}{\mathrm{BW}} \;+\; 2\sqrt{\frac{(L - 1)\,\alpha\,M}{\mathrm{BW}}}$$

The square-root correction grows as $\sqrt{M}$ — much slower than the original $LM/\mathrm{BW}$ or the floor $M/\mathrm{BW}$. The punch line:

> **In the bandwidth-bound regime, pipelining collapses the BW coefficient from $L$ down to (essentially) 1, at a modest $O(\sqrt{M})$ latency-term surcharge. The base $L\alpha$ hop-count cost is untouched.**

This is the mechanism by which a schedule with $L \sim \log N$ steps (double binary tree) can reach near-ring bandwidth despite its pure-model $\log_2 N \cdot M/\mathrm{BW}$ BW term.

**The caveat that makes or breaks the whole thing.** The diagram above is a logical picture — "row $k$" is "step $k$ of the schedule", and each cell shows which segment is being *logically processed* there. To turn this into real wall-clock time, every cell in the steady-state column must correspond to a *physical transfer on a distinct link*. If multiple cells in the same column try to drive the **same physical port** on the same rank, only one of them actually runs; the others queue, and the pipeline collapses partway back to serial.

Counting concurrency at a single rank $R$: during steady state, $R$ is simultaneously participating in all $L$ steps. Each step corresponds to one edge out of $R$ — so $R$ needs **$L$ concurrent physical ports** to sustain the pipeline fully. "Port" here is deliberately generic: whichever level of the fabric the collective traverses, the budget is set by the number of distinct physical links out of $R$ at *that* level. Concretely, the tight budget can be any of:

- **Scale-up on-package or on-node**: NVLink lanes / ring-direction channels out of a GPU (typically 2 ring directions or 4–18 NVLink channels per chip), on-package chiplet-to-chiplet interconnect ports, or PCIe lanes to a host switch — the scale-up fabric that stays inside a high-bandwidth island.
- **Scale-up switch ports**: the upstream ports a single endpoint presents into an NVSwitch / UALink / proprietary scale-up switch tier. A GPU typically connects with a handful of links into that switch, and the switch itself has a fixed radix.
- **Scale-out off-node**: NIC count per GPU or per node (typically 2–8 NICs on modern training nodes), which sets the per-rank port budget for inter-node collectives.
- **Mesh / torus fabrics**: on a $k$-dimensional torus each rank has exactly $2k$ neighbor ports (2 for a 1-D ring, 4 for 2-D, 6 for 3-D, etc.), fixed by the *topology* rather than by the endpoint. This is the tightest and least flexible port budget of any fabric family: an algorithm needing $\log_2 N$ concurrent partners has no hope on a 3-D torus, and even ring-family algorithms have to be carefully dimension-decomposed to stay inside the budget.

Torus-specific consequences — dimension-decomposed ring AR on torus, and the full mapping from the algorithms here to each topology — are worked through in `02_topology_mapping.md`.

Whatever level applies, the number is $O(1)$-to-single-digit, not $O(\log N)$. Algorithms whose steady-state concurrency exceeds that number serialize on the oversubscribed port regardless of whether the oversubscription is inside a scale-up switch, on a torus neighbor link, on the scale-out NIC, or across on-package links.

```
Case A — schedule's per-rank port requirement fits the budget:

  slot 3 (steady state at rank R):
    step 1 edge ─────▶  port A     carrying seg3
    step 2 edge ─────▶  port B     carrying seg2
    step 3 edge ─────▶  port C     carrying seg1
                  └── 3 distinct links, 3 segments in parallel ✓
                      conveyor runs as drawn


Case B — schedule's per-rank port requirement exceeds the budget:

  slot 3 (steady state at rank R, suppose only port A exists):
    step 1 edge ───┐
    step 2 edge ───┼───▶  port A   ← only one gets served per slot
    step 3 edge ───┘                  other two queue behind it
                  └── pipeline collapses toward serial L·(α + M/BW)
```

The central question for any algorithm is therefore: **does the schedule's per-rank concurrency fit the fabric's port budget at every tier the collective traverses?** For ring ($L$ steps but only the 2 immediate-neighbor links used throughout) the answer is yes — 2 ports is the universal minimum, satisfied on both scale-up and scale-out tiers. For double binary tree ($L = 2\lceil\log_2 N\rceil$ steps, but a rank's role switches between the two trees, maxing out at ~3 concurrent partners) the answer is also yes on any node with 3+ concurrent links at the relevant tier. For plain recursive doubling and Rabenseifner ($\log_2 N$ distinct partners across the $\log_2 N$ steps, all needing concurrency to pipeline — 9 partners at $N = 512$), the answer on a typical node (2 NVLink directions, 2–8 NICs) is **no**. The next paragraphs walk through each algorithm's answer in detail.

**Ring is intrinsically pipelined with $P = N$.** The $2(N{-}1)$ steps and $M/N$-byte chunks from §3.1 are not a non-pipelined form — they **are** the pipelined schedule with $P = N$. Every slot, rank $i$ is simultaneously sending a chunk on its right-neighbor link (downstream) and receiving a different chunk on its left-neighbor link (upstream). Two concurrent transfers, but on two distinct physical links, so no conflict. Plugging $L = 2(N{-}1)$ and $P = N$ into the pipelining formula recovers exactly the $2(N{-}1)\alpha + 2(N{-}1)M/(N\,\mathrm{BW})$ entry in the table. Finer segmentation ($P > N$) cannot improve this: the bottleneck neighbor-link already carries $2(N{-}1)M/N$ total bytes across the collective and is saturated every slot. Ring is already at its asymptote.

**Double binary tree is pipeline-friendly because segments at different tree depths use different physical edges.** In the §3.2 construction, each of the two trees spans the $N$ ranks with $N{-}1$ edges, and within one tree the edges at depth 1 are disjoint from the edges at depth 2, and so on. So segment $s$ at step $k$ (traveling across a depth-$k$ edge) and segment $s{+}1$ at step $k{-}1$ (depth $k{-}1$ edge) occupy disjoint physical links — the conveyor runs. Concurrency per rank stays small: the busiest role is a tree's root at the final reduce step, concurrently receiving from its (up to 2) children while also participating in the sibling tree's broadcast as a leaf — 3 concurrent partners at worst, well within the port budget of any modern fabric tier (3+ NVLink channels on-node, 3+ scale-up switch uplinks, or 3+ NICs off-node). Plugging $L = 2\lceil \log_2 N \rceil$ and per-segment payload $M/(2P)$ (since each tree carries half the message) into the pipelining formula:

$$t_{\mathrm{DBT,pipe}}(P^{*}) = 2\lceil \log_2 N \rceil\,\alpha + \frac{c\,M}{\mathrm{BW}} + O\!\left(\sqrt{\log N \cdot \alpha\,M/\mathrm{BW}}\right)$$

for a small constant $c$ of order $1$ (the exact value depends on the tree's edge-disjointness and how saturated each full-duplex link stays across the schedule). **The pure-model $\log_2 N$ factor in the BW term is gone** — replaced by the constant $c$ — while the latency term stays at $O(\log N)\alpha$ plus the $\sqrt{M}$ pipeline-fill correction. That is the "log-depth latency AND near-ring bandwidth" combination that makes double tree the shipping choice for the latency-bound / medium-$M$ regime.

**Plain recursive doubling defeats the conveyor because each step uses a different partner.** At step $k$, rank $i$ exchanges with partner $i \oplus 2^{k-1}$; the partner changes every step. Pipelining $P$ segments means that, once the pipeline is full, rank $i$ is simultaneously mid-exchange with segments at every one of the $\log_2 N$ steps — each targeting a **different partner**. Sustaining that requires $\log_2 N$ concurrent communication ports per rank ($= 9$ for $N = 512$), far beyond the port budget at any relevant fabric tier (2 NVLink ring directions on-node, 2–8 NICs off-node, and typically a single-digit number of scale-up switch uplinks per endpoint). On the real fabric only a handful of those concurrent streams can proceed; the remainder serialize. The schedule effectively degrades back to the non-pipelined $\log_2 N(\alpha + M/\mathrm{BW})$, and the $\log_2 N$ BW coefficient survives. Segmenting alone can't save it because each of the $\log_2 N$ steps requires the **full accumulated buffer** — the partial sum over $2^k$ ranks after step $k$ — so segmentation shrinks per-segment bytes without reducing the per-step volume one rank must push to its current partner. Both issues compound; either alone would rule out pipelined recursive doubling.

**Rabenseifner has the same partner-cycling problem.** Both the RS phase (recursive-halving) and AG phase (recursive-doubling) cycle through $\log_2 N$ partners. Pipelining again demands $\log_2 N$ concurrent ports per rank, so the pipeline serializes on real fabrics and the $2(N{-}1)/N$ BW coefficient of the non-pipelined form is the best it attains. Double tree, starting from a worse non-pipelined $\log_2 N$ BW coefficient, matches or beats that asymptote via pipelining — which is why double tree wins the shipping slot despite Rabenseifner's better-looking table entry.

**Two secondary factors reinforce the choice.** First, **arbitrary $N$**: plain recursive doubling and the cleanest Rabenseifner form require $N = 2^k$; cluster collective sizes — set by the job's parallelism configuration — are rarely pure powers of two. Ring handles any $N$; double tree handles any $N$ via a mildly asymmetric tree construction. Second, **topology fit**: ring embeds onto any fabric with a logical cycle (torus wraparound, NVLink ring on-node), while double tree prefers fat-tree / Clos fabrics with plentiful bisection — together covering the deployment landscape of modern GPU clusters. Pipelining on multi-tier fabrics is revisited in `02_topology_mapping.md`.

**Net effect on latency and bandwidth.** For algorithms whose schedule is compatible with a small-port fabric (ring, double tree): pipelining **drops the BW coefficient from whatever the non-pipelined form had ($L$ for double tree's straight-line-per-tree schedule) down to $O(1)$**, at the cost of a modest $O(\sqrt{M})$ latency overhead and no change to the $L\alpha$ base. For partner-cycling algorithms (plain recursive doubling, Rabenseifner): pipelining cannot overlap on a bounded-port fabric, so **neither term improves meaningfully**. The gap between the pure-model table and the shipping choice is almost entirely this structural asymmetry.

**Numerical crossovers.** Equate any two rows in the cost table and solve for $M$ to get the algorithm crossover $M_*$. At representative scale-up fabric parameters ($\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$ — NVLink-5 / NVSwitch class):

- **Rec-doub vs DBT:** $M_* = \lceil \log_2 N \rceil \cdot \alpha \cdot \mathrm{BW} / (\lceil \log_2 N \rceil - 2)$. Below $M_*$, rec-doubling wins on fewer α hops; above, DBT's constant $2 M/\mathrm{BW}$ beats rec-doubling's $\log_2 N \cdot M/\mathrm{BW}$. Numerically: $N = 72 \Rightarrow M_* \approx 630\,\mathrm{KB}$; $N = 512 \Rightarrow M_* \approx 580\,\mathrm{KB}$. Any realistic gradient / activation AR ($M \gtrsim$ a few MB) is firmly in DBT's regime.
- **Ring vs DBT:** $M_* = N \cdot (N - 1 - \lceil \log_2 N \rceil) \cdot \alpha \cdot \mathrm{BW}$. At $N = 72$: $M_* \approx 2\,\mathrm{GB}$; at $N = 512$: $M_* \approx 116\,\mathrm{GB}$. DBT dominates ring across every practical payload once the fabric has any-to-any routing.
- **Rabenseifner vs DBT:** asymptotically tied in cost ($2(N-1)/N \to 2$). DBT wins on non-cost grounds covered above — pipeline feasibility, any-$N$ support, single-kernel schedule.

Takeaway under the pure pipelined α-β model: DBT is the right default for all non-tiny AR on any-to-any fabrics; rec-doubling is reserved for sub-MB control messages; ring's α cost is prohibitive across the entire realistic $M$ range. **Practice caveat.** Production runtimes measure a different crossover: NCCL's tuner picks DBT for small-$M$ AR and ring for large-$M$ AR, and published benchmarks [DEMYST-NCCL] confirm this inversion. The pipelining argument above sets an *upper bound* on what DBT's BW coefficient can be — the constant $c$ in the DBT pipelined formula is $\geq 1$ in practice, while ring's is already at the $2(N-1)/N \to 2$ floor. Tree-specific implementation overhead (finite pipeline depth $P \ll P^{*}$, per-step kernel complexity, CUDA launch and synchronization granularity) pushes $c$ above ring's 2 at bulk $M$, so ring re-takes the crown past a runtime-calibrated crossover. The α-β-only reasoning remains the right intuition for small-to-medium $M$ where DBT's lower $\alpha$ dominates; ring wins when BW overhead per step is small enough that its $O(N)$ α cost amortizes into the pipeline.

---

## 4. All-gather / reduce-scatter

AG and RS are the two halves of AR and also useful primitives in their own right. **AG** starts with each rank holding one $M/N$-byte chunk and ends with all ranks holding the concatenation of all $N$ chunks. **RS** is the dual: each rank starts with the full $M$-byte vector and ends holding one $M/N$ chunk reduced across all ranks. Both appear pervasively in sharded-parameter training (gradient RS into sharded optimizer state, weight AG before each forward pass), sequence-sharded activations, and any scheme that splits a tensor across ranks and needs to rematerialize or reduce it.

Like AR, both primitives have ring-based (§4.1) and tree-flavored implementations with similar latency-vs-bandwidth tradeoffs. NCCL ships ring AG / RS as the default for both primitives. Two alternate variants appear in the literature — recursive-doubling AG / recursive-halving RS from the MPI menu (full derivation in [Appendix B.1](#b1-recursive-doubling-ag--recursive-halving-rs)), and the more recent **Parallel Aggregated Trees (PAT)** algorithm introduced by NCCL 2.23 specifically for scale-out AG / RS at 1 rank per node (full derivation in [Appendix B.2](#b2-parallel-aggregated-trees-pat--scale-out-ag--rs)). §4.2 summarizes the comparison and explains the NCCL selection rule, with cross-refs into the appendix for the mechanics.

### 4.1 Ring-based AG / RS

Ring-based AG is exactly Phase 2 of ring AR from §3.1; ring-based RS is exactly Phase 1. As standalone collectives they each take $N-1$ steps:

$$t_{\mathrm{ring\,RS}} = t_{\mathrm{ring\,AG}} = (N-1)\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

where $M$ is the **per-rank final total volume** — for AG, the size each rank ends with; for RS, the size each rank starts with before the reduce. The bandwidth term is exactly half of ring AR's because you only do one of the two phases.

The pipelining analysis from §3.3 carries over directly: the ring's two-neighbor-link schedule already saturates both directions of the adjacent links, so the formula is at its asymptotic minimum and finer segmentation gives no further gain.

### 4.2 Comparison and NCCL selection

| Algorithm | Latency | Bandwidth | NCCL adoption |
|---|---|---|---|
| Ring (§4.1) | $(N-1)\,\alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Y (all regimes; scale-up and scale-out default) |
| Recursive doubling AG / halving RS ([App. B.1](#b1-recursive-doubling-ag--recursive-halving-rs)) | $\lceil \log_2 N \rceil \, \alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | N (MPI / OpenMPI menu; NCCL does not ship) |
| PAT ([App. B.2](#b2-parallel-aggregated-trees-pat--scale-out-ag--rs)) | $\lceil \log_2 N \rceil \, \alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Y (NCCL 2.23+, **inter-node only, 1 rank per node**, scale-out) |

**The BW term is identical for all three algorithms** — each ships a total of $(N-1)M/N$ bytes per rank, just in different round structures. The only cost difference is the α term: $\lceil \log_2 N \rceil$ vs $N-1$. Tree-flavored AG / RS therefore cost strictly less than ring on α-β arithmetic, yet NCCL's default path remains ring for all scale-up use and most scale-out use — and where PAT does ship (scale-out, 1 rank/node), its partner schedule is specifically engineered around the scale-out fabric's bottleneck.

**Why NCCL ships ring for scale-up and for most scale-out AG / RS.** The reasons mirror the DBT-vs-Rabenseifner argument for AR in §3.3, and come from the same port-budget / partner-cycling analysis:

- **Pipeline feasibility.** Ring's per-round chunk is a constant $M/N$, so chunks stream through the ring at wire speed — the α term amortizes across the pipeline and vanishes at large $M$. Recursive-doubling AG ships chunks of $M/N, 2M/N, 4M/N, \ldots$ — geometrically growing — so a pipelined schedule needs custom per-round chunk geometry, or drains the pipeline at every round boundary. Recursive-halving RS has the mirror problem (chunks halving each round).
- **Any $N$.** Rec-doubling / rec-halving strictly need $N = 2^k$ for clean bit-pairings. Ring handles any $N$, including the odd sizes that fall out of pipeline-parallel stage assignments or irregular job configurations.
- **Kernel simplicity.** Ring is a single communication loop with fixed chunk size; rec-doubling is $\lceil \log_2 N \rceil$ distinct rounds, each with a different partner offset and chunk size.
- **Overlap with compute.** The steady-chunk pattern of ring AG overlaps naturally with per-chunk compute (e.g., FSDP / ZeRO gather-then-compute). Rec-doubling's round boundaries are global sync points that break overlap.

MPI (MPICH, OpenMPI) does ship rec-doubling AG and rec-halving RS as algorithm options, typically dispatched by message-size threshold at runtime; NCCL's single-algorithm-per-primitive default prioritizes the pipelineable ring case.

**Why PAT ships for scale-out AG / RS at 1 rank/node and not for scale-up.** PAT (Parallel Aggregated Trees) [NCCL-PAT] addresses the specific regime where the scale-out fabric is the bottleneck — one rank per node communicating over the NIC / inter-node path — and the ring's $(N-1)\alpha$ term becomes prohibitive once $N$ is the node count rather than the intra-node GPU count. PAT reverses the Bruck offset schedule (largest hops first), which keeps the long-distance network transfers to a small number of logarithmic rounds rather than $N-1$ sequential neighbor hops, and is built around a **bounded intermediate buffer** so it composes with the inter-node path's finite staging memory. Two constraints limit where it ships:

1. **Inter-node only, 1 rank per node.** The NCCL 2.23 implementation restricts PAT to one rank per node because only the inter-node phase is implemented; intra-node AG / RS still runs ring. This matches the scale-out design intent — where PAT's log-depth structure pays off against the scale-out fabric's α cost, not against on-node NVLink latency.
2. **Scale-up doesn't benefit.** On a scale-up NVLink / NVSwitch domain, ring's α cost is already small (α ≈ 0.5 μs × $N < 72$ = a few tens of μs), and ring's BW-optimal pipelining keeps it at the BW floor. Replacing it with PAT would trade pipeline-friendliness for a log-depth α schedule that doesn't fit the scale-up port budget any better than rec-doubling does — the same partner-cycling objection from AR §3.3 applies.

The scale-out motivation — and the specific "reversed Bruck + bounded buffer" mechanism that PAT uses to make the partner cycling fit the scale-out fabric — is worked out step-by-step at $N = 8$ in [Appendix B.2](#b2-parallel-aggregated-trees-pat--scale-out-ag--rs).

**Picking between ring, rec-doub / rec-halv, and PAT.**

| Variant | Cost | Regime where NCCL selects it |
|---|---|---|
| Ring AG / RS (§4.1) | $(N-1)\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$ | All scale-up AG / RS; scale-out AG / RS when $N$ small or multiple ranks per node (NCCL default) |
| Rec-doub AG / rec-halv RS (App. B.1) | $\lceil \log_2 N \rceil \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$ | Not selected by NCCL; MPI-only |
| PAT (App. B.2) | $\lceil \log_2 N \rceil \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$ | Scale-out AG / RS at 1 rank/node, where ring's $(N-1)\alpha$ across the inter-node fabric dominates |

There is no α-β cost crossover between the three — BW terms are all equal, so rec-doubling and PAT both beat ring on α for every $(N, M)$. Ring remains the pipelineable-and-any-$N$ default for scale-up and most scale-out use; PAT wins where the inter-node α cost matters and the bounded-buffer structure is the right fit; rec-doubling / rec-halving's partner-cycling failure on bounded-port fabrics keeps it out of the NCCL menu entirely.

---

## 5. All-to-all

A2A permutes data: rank $i$ holds $N$ chunks where chunk $j$ is destined for rank $j$; after the collective, rank $i$ holds $N$ chunks where chunk $j$ was contributed by rank $j$. The transpose pattern is a pure permutation — no summation, no reduction — so aggregate cross-fabric traffic is exactly $(N{-}1)M$ bytes (each of the $N$ ranks ships $(N{-}1)/N$ of its per-rank payload to other ranks). This makes A2A the **most bandwidth-hungry** of the primitives in this note: unlike AR, whose RS+AG decomposition compresses the BW term to $2(N{-}1)/N \cdot M/\mathrm{BW}$, A2A has no such decomposition to hide behind.

**Worked example at $N=4$.** Each rank $R_i$ starts with $N$ chunks $v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}$, where $v_{i,j}$ is the chunk that $R_i$ contributes *to* $R_j$. The transpose pattern arises in any permutation-shaped redistribution — MoE expert dispatch, distributed FFTs, matrix transposes, shuffle-in-parallel sorting — and the derivation below is agnostic to which.

```
Initial (each rank holds 4 chunks, one per destination):

R0: [v00 → R0   v01 → R1   v02 → R2   v03 → R3]
R1: [v10 → R0   v11 → R1   v12 → R2   v13 → R3]
R2: [v20 → R0   v21 → R1   v22 → R2   v23 → R3]
R3: [v30 → R0   v31 → R1   v32 → R2   v33 → R3]
```

After A2A, each rank holds the column of chunks destined for it:

```
Final state:

R0: [v00  v10  v20  v30]    ← all chunks whose destination was R0
R1: [v01  v11  v21  v31]
R2: [v02  v12  v22  v32]
R3: [v03  v13  v23  v33]
```

The (source, destination) matrix is transposed along the diagonal. §5.1 covers the pairwise direct-send schedule that NCCL ships — a ring-equivalent in the α-β model that hits the bandwidth lower bound at $(N{-}1)\alpha$ latency — and §5.2 compares it to the Bruck $O(\log N)$-latency alternative (derivation in [Appendix B.3](#b3-bruck-a2a)) and explains the NCCL selection rule.

### 5.1 Pairwise direct-send A2A

On a logical ring, A2A runs as a relay: at step $t \in \{1, \ldots, N{-}1\}$, rank $i$ sends the chunk destined for rank $(i + t) \bmod N$ to its right neighbor; the receiver either keeps the chunk (if it is the destination) or forwards it next step. On a fabric with sufficient bisection bandwidth (fat-tree / Clos / NVSwitch), the equivalent **pairwise direct-send** variant sends each chunk straight from source to destination — same step count, same cost in the α-β model, no intermediate relay. **NCCL ships the pairwise direct-send variant** via its staggered P2P scheduler (`scheduleP2pTasksToPlan`), which offsets the per-rank partner order by rank index so that step 1 routes $R_0 \to R_1, R_1 \to R_2, \ldots$ (adjacent offsets), step 2 routes offset $+2$, and so on — spreading concurrent sends across distinct switch-port pairs and avoiding per-step head-of-line blocking on any single port. We trace the pairwise-direct version below since it makes the chunk movement easiest to see.

**Worked example at $N=4$.** At step $t \in \{1, 2, 3\}$, rank $R_i$ ships $v_{i, (i+t) \bmod 4}$ to $R_{(i+t) \bmod 4}$ and concurrently receives $v_{(i-t) \bmod 4,\, i}$ from $R_{(i-t) \bmod 4}$ over the full-duplex link. Each rank does exactly one send + one receive per step.

Starting state (repeated from §5 intro for convenience):

```
R0: {v00, v01, v02, v03}
R1: {v10, v11, v12, v13}
R2: {v20, v21, v22, v23}
R3: {v30, v31, v32, v33}
```

**After step 1** (offset $+1$) — each rank ships its "right-neighbor" chunk and receives from its "left-neighbor".

```
Sends:  R0→R1: v01    R1→R2: v12    R2→R3: v23    R3→R0: v30

R0: {v00,      v02, v03,  v30}    ← received v30 from R3
R1: {v01, v10, v11,      v13}     ← received v01 from R0
R2: {     v12, v20, v21, v22}     ← received v12 from R1
R3: {          v23, v31, v32, v33}    ← received v23 from R2
```

Each rank now holds 2 of its 4 final chunks (its own self-chunk plus one received).

**After step 2** (offset $+2$) — each rank ships its "2-away" chunk.

```
Sends:  R0→R2: v02    R1→R3: v13    R2→R0: v20    R3→R1: v31

R0: {v00,           v03,  v20, v30}    ← received v20 from R2
R1: {v01, v10, v11,             v31}   ← received v31 from R3
R2: {     v02, v12,      v21, v22}     ← received v02 from R0
R3: {          v13, v23,      v32, v33}    ← received v13 from R1
```

**After step 3** (offset $+3$) — each rank ships its last remaining foreign chunk.

```
Sends:  R0→R3: v03    R1→R0: v10    R2→R1: v21    R3→R2: v32

R0: {v00, v10, v20, v30}    ← received v10 from R1
R1: {v01, v11, v21, v31}    ← received v21 from R2
R2: {v02, v12, v22, v32}    ← received v32 from R3
R3: {v03, v13, v23, v33}    ← received v03 from R0
```

Every rank now holds its column of the transpose. A2A done.

**Cost accounting.** Each of the 3 steps is one send + one receive per rank over a single link, each carrying $M/N = M/4$ bytes. The 3 steps are sequential (every rank is busy each step, talking to a different partner), so costs add:

$$t_{\mathrm{ring\,A2A}}\bigg|_{N=4} = 3\alpha + 3 \cdot \frac{M}{4\,\mathrm{BW}}$$

Generalizing to $N$ ranks:

$$t_{\mathrm{ring\,A2A}} = (N-1)\,\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

This hits the $(N{-}1)/N \cdot M/\mathrm{BW}$ lower bound for A2A and is pipeline-friendly: each rank uses only its two adjacent links (ring relay) or one outbound + one inbound link per step (pairwise direct), well within any fabric's port budget.

Workloads that run A2A back-to-back (e.g., MoE dispatch followed by a reverse A2A combine) double the total, giving $2(N{-}1)\alpha + 2(N{-}1)M/(N\,\mathrm{BW})$ for the round trip.

### 5.2 Comparison and NCCL selection

| Algorithm | Latency | Bandwidth | NCCL adoption |
|---|---|---|---|
| Pairwise direct-send (§5.1) | $(N-1)\,\alpha$ | $\dfrac{N-1}{N} \cdot \dfrac{M}{\mathrm{BW}}$ | Y |
| Bruck ([App. B.3](#b3-bruck-a2a)) | $\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot \dfrac{M}{2\,\mathrm{BW}}$ | N |

Two features stand out compared to §3.3 and §4.2:

- **A2A has no bandwidth-optimal log-depth algorithm.** The DBT-style construction that gave AR both log-depth latency and near-ring bandwidth does not transfer. The practitioner's choice is between pairwise's BW-optimal but $O(N)$-latency schedule and Bruck's $O(\log N)$-latency but BW-inflating schedule, with a crossover at $M^{*} \sim \alpha \cdot \mathrm{BW}$ — but no "best of both worlds" option.
- **A2A is the collective most likely to be bandwidth-bound on a bisection-constrained fabric.** The aggregate $(N{-}1)M$ cross-fabric traffic has to pass through the bisection of whatever physical topology the $N$ ranks sit on. On topologies with limited bisection (torus), this bound turns A2A into the rate-limiting step for permutation-heavy workloads (MoE, distributed FFT, distributed sort) — and drives physical-layout decisions about which ranks sit inside which high-BW island. The topology discussion in `02_topology_mapping.md` makes this quantitative.

**Why NCCL ships pairwise over Bruck.**

- **Bandwidth optimality wins at large $M$.** Pairwise's $(N-1)/N \cdot M/\mathrm{BW}$ is BW-optimal; Bruck's $\log_2 N / 2 \cdot M/\mathrm{BW}$ scales with $\log_2 N$. Above the crossover $M_*$ below, pairwise is strictly faster, and most A2A payloads (MoE dispatch at training batch sizes, activation shuffles, bulk data transposes) are comfortably above $M_*$.
- **Steady chunk size → pipelineable.** Every pairwise round ships a fixed $M/N$ chunk, and chunks stream through the fabric at wire speed. Bruck's per-round chunk contains slots destined for different ranks, so the per-slot accounting is irregular and each round must complete before the buffer-rotation step for the next round can begin.
- **No pre/post-rotation.** Bruck requires cyclic rotation of the send buffer by $-i$ before communication and by $+i$ after — extra local work plus an intermediate buffer of size $M$. Pairwise is zero-copy in the steady state.
- **Any $N$.** Bruck's bit-test routing is cleanest at $N = 2^k$; non-power-of-2 needs padding or a modified final round. Pairwise handles any $N$ directly.
- **Kernel simplicity.** Pairwise is a single loop; Bruck is $\lceil \log_2 N \rceil$ distinct rounds with different partner offsets plus two rotation phases.

MPI variants (MPICH, OpenMPI) ship both and dispatch by message size and $N$ at runtime. Bruck is the MPI default for small $M$; pairwise for large $M$. NCCL's single-algorithm design commits to the large-$M$ case that dominates on-GPU A2A.

**Picking between pairwise and Bruck.**

| Variant | Cost | $n_\alpha$ | Regime where it wins |
|---|---|---|---|
| Pairwise direct-send | $(N-1)\alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$ | $N-1$ | Large-$M$ A2A (MoE dispatch at high batch, bulk shuffles). NCCL default |
| Bruck | $\lceil \log_2 N \rceil \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{2 \cdot \mathrm{BW}}$ | $\lceil \log_2 N \rceil$ | Small-$M$ A2A (control / metadata / handshake patterns) |

Equate costs and solve for $M$: $M_* = \dfrac{(N - 1 - \lceil \log_2 N \rceil) \cdot \alpha \cdot \mathrm{BW}}{\lceil \log_2 N \rceil / 2 - (N-1)/N}$. At representative scale-up fabric parameters ($\alpha = 0.5\,\mu$s, $\mathrm{BW} = 900\,\mathrm{GB/s}$): $N = 72 \Rightarrow M_* \approx 11.5\,\mathrm{MB}$; $N = 512 \Rightarrow M_* \approx 64\,\mathrm{MB}$. Below $M_*$ Bruck wins on α compression; above, pairwise wins on BW optimality.

**Where real workloads sit relative to $M_*$.** The abstract $(N, M)$ maps to workload parameters straightforwardly: $N$ is the A2A communicator size (EP group size for MoE dispatch — equal to $n_\mathrm{experts}$ when one expert is placed per rank — or TP/SP degree for a bulk activation / tensor transpose), and per-rank $M$ scales as (tokens per rank) × fanout × $H$ × (bytes per element), where fanout is the router's top-$k$ for MoE dispatch and $1$ for a plain permutation. Bulk training-step or long-sequence payloads land well above $M_*$ (pairwise regime); sub-MB control-plane A2A lands below (Bruck regime); workloads combining small per-rank token counts with large $N$ — latency-critical dispatch at high expert count — can sit near the crossover, which is the regime where algorithm choice becomes a measurable performance lever rather than a rounding error.

**What the NCCL default leaves on the table.** In the deep-Bruck regime — small per-rank $M$ combined with large $N$, e.g. fine-grained MoE dispatch at low batch — NCCL runs pairwise with $N-1$ α-rounds when $\lceil \log_2 N \rceil$ would do. Two effects compress the realized speedup below the α-β ceiling: the kernel-launch floor ($\sim$5-10 μs on current paths) absorbs part of the theoretical α savings at sub-MB sizes, and Bruck's pre/post buffer rotation is local work that's non-trivial when $M$ is small. The practical gap is therefore smaller than the formula predicts, but it's non-zero and visible in microbenchmarks — so the single-algorithm default is a defensible simplification for production, not a statement that Bruck has nothing to offer.

---

## 6. Point-to-point hop

A single send/recv between two ranks — the degenerate $N=2$ case of any collective, shown for completeness in the same style as §§3–5.

**Worked example.** Sender $R_A$ holds payload $v$ (size $M$ bytes); receiver $R_B$ is empty. One step, one link, one $M$-byte transfer:

```
Initial:
R_A: {v}
R_B: { }

Step 1 — R_A sends v to R_B over the link:

R_A: {v}        ← kept (send is non-destructive)
R_B: {v}        ← received
```

**Cost.** One handshake plus $M$ bytes on a single link:

$$t_{\mathrm{P2P}} = \alpha + \frac{M}{\mathrm{BW}}$$

Trivial, but it shows up once per pipeline step: pipeline parallelism (PP) passes activations stage-by-stage as a chain of P2P hops. If the PP stage count is $P$, the per-step PP cost is $\alpha + M_{\mathrm{PP}} / \mathrm{BW}$ per hop, contributing to the stage-level comm time.

---

## 7. Mapping primitives to DP, TP, EP, SP, PP

Large-model distributed execution uses up to five orthogonal parallelism axes, each mapped to a collective primitive. Four of them (TP/SP/EP/PP) contribute cost **inside every forward / decode step** and dominate inference latency; DP (data parallel) adds a **once-per-training-step** gradient AR that is absent from pure inference but often the largest single collective in training.

| Parallelism | Collective | Purpose | Per-step message size |
|---|---|---|---|
| **DP** (data parallel) | **AR** | Reduce gradients across DP replicas (training only) | $M_{\mathrm{DP}}$ = full gradient vector per AR call; amortized once per training step, not per token |
| **TP** (tensor parallel) | **AR** | Reduce partial outputs of row-sharded attention / MLP projections | $M_{\mathrm{TP}} = 2 b H / (TP \cdot SP)$ (per token, per layer) |
| **SP** (sequence parallel) | **AG** | Gather KV cache shards across sequence chunks | $M_{\mathrm{SP}}$ depends on KV shard size |
| **EP** (expert parallel) | **A2A** | Dispatch / combine MoE-routed tokens | $M_{\mathrm{EP}} = k H b$ per token routed off-rank |
| **PP** (pipeline parallel) | **P2P** | Forward activations between stages | $M_{\mathrm{PP}} = b H$ per token |

A single decoding step on a model sharded across TP/SP/EP/PP issues one AR per TP layer, one AG per SP layer, one A2A per MoE layer (dispatch + combine = two A2A), and one P2P per pipeline hop. The aggregate per-stage communication time is

$$t_{\mathrm{comm,stage}} = \frac{L}{PP}\,(n_{\mathrm{TP}} \cdot t_{\mathrm{TP}} + n_{\mathrm{SP}} \cdot t_{\mathrm{SP}}) + \frac{L_{\mathrm{MoE}}}{PP} \cdot n_{\mathrm{EP}} \cdot t_{\mathrm{EP}} + t_{\mathrm{PP}}$$

where $n_{\mathrm{TP}}, n_{\mathrm{SP}}, n_{\mathrm{EP}}$ are the number of collectives per layer (typically 2, 2, 2 for LLaMA-style blocks). This is the single-pass (decode) aggregation; the training-side analog adds backward-pass AG/RS per layer plus a DP gradient AR per step that typically uses ring or Rabenseifner from §3 at the full-gradient-vector scale.

### Vocabulary aside: why these mappings and not others?

- **DP → AR** (not A2A or AG) because DP replicas each compute gradients on different mini-batches and must **sum** them to get the true batch gradient. Same logic as TP's partial-output reduction, just at a different granularity (whole-gradient vs. per-layer partial).
- **TP → AR** (not AG) because TP splits weight matrices across ranks; each rank computes a **partial** output that must be **summed** to produce the true output. RS+AG is the bandwidth-optimal way to do that sum.
- **SP → AG** (not AR) because SP splits tokens across ranks; each rank holds **distinct** partial KV and just needs to share it, no reduction.
- **EP → A2A** (not AR) because MoE routing is a **permutation** — token $t$ belongs to whichever expert the router picked; it needs to physically move there. No summation.
- **PP → P2P** (not a collective) because PP is a chain: the activation at stage $s$ only goes to stage $s+1$. No group operation.

---

## 8. Cost summary table

Complete cost table for the primitives above on an idealized fully-connected fabric (no contention, uniform BW, per-hop $\alpha$):

| Primitive | Algorithm | Latency term | BW term | NCCL adoption |
|---|---|---|---|---|
| AR | Ring (§3.1) | $2(N-1)\,\alpha$ | $2 \cdot (N-1)/N \cdot M/\mathrm{BW}$ | Y |
| AR | Double binary tree (§3.2) | $2\,\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | Y |
| AR | Simple rec. doubling (App. A.1) | $\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | N |
| AR | Rabenseifner (App. A.2) | $2\,\lceil \log_2 N \rceil \, \alpha$ | $2 \cdot (N-1)/N \cdot M/\mathrm{BW}$ | N |
| AG | Ring (§4.1) | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | Y |
| AG | PAT (App. B.2) | $\lceil \log_2 N \rceil \, \alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | Y (inter-node only, 1 rank/node) |
| AG | Recursive doubling (App. B.1) | $\lceil \log_2 N \rceil \, \alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | N |
| RS | Ring (§4.1) | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | Y |
| RS | PAT (App. B.2) | $\lceil \log_2 N \rceil \, \alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | Y (inter-node only, 1 rank/node) |
| RS | Recursive halving (App. B.1) | $\lceil \log_2 N \rceil \, \alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | N |
| A2A (dispatch only) | Pairwise direct-send (§5.1) | $(N-1)\,\alpha$ | $(N-1)/N \cdot M/\mathrm{BW}$ | Y |
| A2A (dispatch+combine) | Pairwise direct-send (§5.1) | $2(N-1)\,\alpha$ | $2(N-1)/N \cdot M/\mathrm{BW}$ | Y |
| A2A (dispatch only) | Bruck (App. B.3) | $\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot M/(2\,\mathrm{BW})$ | N |
| A2A (dispatch+combine) | Bruck (App. B.3) | $2\,\lceil \log_2 N \rceil \, \alpha$ | $\lceil \log_2 N \rceil \cdot M/\mathrm{BW}$ | N |
| P2P | direct (§6) | $\alpha$ | $M/\mathrm{BW}$ | Y |

Two observations that propagate into every topology discussion:

1. **Ring and tree trade latency for bandwidth.** Ring is bandwidth-optimal ($M/\mathrm{BW}$ coefficient can't drop below $(N-1)/N$); tree is latency-optimal ($\log_2 N$ hops is a lower bound on any reduction schedule over a binary-combinable tree). The hardware picks whichever minimizes $t$ at the given $(N, M, \alpha, \mathrm{BW})$.
2. **A2A is more expensive than AR at fixed $M$.** AR's bandwidth term caps at $2M/\mathrm{BW}$ because the RS+AG decomposition only ships one copy of each chunk; A2A has to ship *everything* from everyone to everyone. On bisection-constrained topologies (torus), this asymmetry determines the layout choice for MoE models.

---

## Appendix A: Non-mainline AR variants

The AR variants below — simple recursive-doubling AR and Rabenseifner halving-doubling AR — appear in the MPI literature and in MPICH / OpenMPI's algorithm menu, and each has a clean pure-model cost that looks competitive on paper. Neither is in the NCCL / RCCL shipping menu: the port-budget and partner-cycling arguments in §3.3 explain why both collapse to worse-than-table costs on real bounded-port fabrics. We derive them at $N=4$ so the comparison in §3.3 has concrete schedules to cite and so the reader can verify the cost formulas on their own. The ring AR at §3.1 and the double binary tree at §3.2 remain the two algorithms NCCL actually selects between.

### A.1 Simple recursive-doubling AR

Simple recursive-doubling AR is the one-phase butterfly / hypercube sweep: at step $k \in \{1, \ldots, \lceil \log_2 N \rceil\}$, every rank $i$ exchanges its **full current vector** with partner $i \oplus 2^{k-1}$ and sums the received vector into its local copy. After $\lceil \log_2 N \rceil$ steps, every rank holds the $N$-way sum. We walk through $N=4$; same initial state as §3.1 (each $R_i$ holds $V_i = [v_{i,0}, v_{i,1}, v_{i,2}, v_{i,3}]$).

**Step 1 (partner $i \oplus 1$, offset $2^0 = 1$).** Pairs: $(R_0, R_1)$ and $(R_2, R_3)$. Each pair exchanges full vectors and sums.

```
After step 1 (2-way partial sums):

R0: [v00+v10   v01+v11   v02+v12   v03+v13]
R1: [v00+v10   v01+v11   v02+v12   v03+v13]   ← identical to R0
R2: [v20+v30   v21+v31   v22+v32   v23+v33]
R3: [v20+v30   v21+v31   v22+v32   v23+v33]   ← identical to R2
```

**Step 2 (partner $i \oplus 2$, offset $2^1 = 2$).** Pairs: $(R_0, R_2)$ and $(R_1, R_3)$. Each pair again exchanges full vectors and sums — but now the "full vector" is already a 2-way partial sum from step 1.

```
After step 2 (4-way full sums):

R0: [S0   S1   S2   S3]
R1: [S0   S1   S2   S3]
R2: [S0   S1   S2   S3]
R3: [S0   S1   S2   S3]

where S_k = v_{0,k} + v_{1,k} + v_{2,k} + v_{3,k}.
```

All four ranks hold the complete reduced vector after $\lceil \log_2 4 \rceil = 2$ sequential steps.

**Cost.** Each step moves the **full** $M$-byte vector across the active link (full-duplex, so both partners send concurrently over opposite directions; per-step cost is $\alpha + M/\mathrm{BW}$). With $\lceil \log_2 N \rceil$ sequential steps:

$$t_{\mathrm{rec\,doubling\,AR}} = \lceil \log_2 N \rceil \, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{\mathrm{BW}}$$

**Strengths.** Minimum latency term $\lceil \log_2 N \rceil \alpha$ — a lower bound for any reduction over a binary-combinable tree. Single-phase schedule (no separate RS / AG split), so the runtime is simpler than Rabenseifner's.

**Weakness.** The BW coefficient grows as $\log_2 N$ rather than saturating at the ring-optimal $2(N-1)/N \to 2$. At $N = 512$, that's $9 \cdot M/\mathrm{BW}$ versus ring's $\approx 2 \cdot M/\mathrm{BW}$ — a $4.5\times$ BW penalty that kills it on large-$M$ AR. The §3.3 pipelining analysis further explains why this BW coefficient **cannot be rescued by segmentation** on a bounded-port fabric: each step targets a different partner, so a pipelined schedule would need $\lceil \log_2 N \rceil$ concurrent physical ports per rank, which no real scale-up or scale-out fabric provides.

### A.2 Rabenseifner halving-doubling AR

Rabenseifner's halving-doubling AR (RHD) [TRG05] recognizes the AR ≡ RS + AG identity from §2 and builds each phase from a chunk-exponential hypercube schedule: the RS phase halves the per-step payload, the AG phase doubles it, and together the two phases ship only $2(N-1)/N \cdot M$ bytes per rank — matching ring's BW-optimal lower bound at only $2\lceil \log_2 N \rceil$ steps. We trace $N=4$ with the same $V_i$ as §3.1, chunked into 4 pieces.

**Phase 1 — Reduce-scatter (recursive halving, $\lceil \log_2 N \rceil$ steps).**

At step $k \in \{1, \ldots, \lceil \log_2 N \rceil\}$, rank $i$ pairs with partner $i \oplus 2^{k-1}$, sends the half of its chunks the partner will own at the end of this step, and receives (summing in) the complementary half. The per-step payload **halves** each round because the "chunks I still own" set halves.

**Step 1 ($k=1$, partner $i \oplus 1$).** Split the 4-chunk vector into lower-half $\{0, 1\}$ and upper-half $\{2, 3\}$. Pairs: $(R_0, R_1)$ → $R_0$ keeps lower, $R_1$ keeps upper. $(R_2, R_3)$ → $R_2$ keeps lower, $R_3$ keeps upper. Each rank sends $M/2$ bytes.

```
After RS step 1:

R0: [v00+v10   v01+v11    ?         ?      ]   (owns {0,1}; sent {2,3} to R1)
R1: [   ?         ?     v02+v12  v03+v13  ]   (owns {2,3})
R2: [v20+v30   v21+v31    ?         ?      ]   (owns {0,1})
R3: [   ?         ?     v22+v32  v23+v33  ]   (owns {2,3})
```

**Step 2 ($k=2$, partner $i \oplus 2$).** Each pair subdivides its half again. $(R_0, R_2)$ both own $\{0, 1\}$; $R_0$ keeps chunk $0$, $R_2$ keeps chunk $1$. $(R_1, R_3)$ both own $\{2, 3\}$; $R_1$ keeps chunk $3$, $R_3$ keeps chunk $2$. Each rank sends $M/4$ bytes.

```
After RS step 2 (full 4-way sums in each rank's one owned chunk):

R0: [S0     ?     ?     ?  ]   ← chunk 0 fully reduced
R1: [ ?     ?     ?    S3  ]   ← chunk 3 fully reduced
R2: [ ?    S1     ?     ?  ]   ← chunk 1 fully reduced
R3: [ ?     ?    S2     ?  ]   ← chunk 2 fully reduced
```

This is the RS end-state. Each rank owns exactly one fully-reduced $M/N$ chunk.

**Phase 2 — All-gather (recursive doubling, $\lceil \log_2 N \rceil$ steps).** The same partner schedule in **reverse order**: step 1 uses partner $i \oplus 2$ (last RS partner), step 2 uses partner $i \oplus 1$ (first RS partner). Each step **doubles** the chunks-owned set; payload grows from $M/N$ to $2M/N$, $\ldots$, to $M/2$ by the last step.

**Step 1 ($k=1$ of AG, partner $i \oplus 2$).** $R_0$ sends its $S_0$ to $R_2$; $R_2$ sends its $S_1$ to $R_0$. Pair $(R_1, R_3)$ similarly exchanges $\{S_3, S_2\}$. Each rank sends $M/4$.

```
After AG step 1:

R0: [S0   S1    ?    ?]
R1: [ ?    ?   S2   S3]
R2: [S0   S1    ?    ?]
R3: [ ?    ?   S2   S3]
```

**Step 2 ($k=2$ of AG, partner $i \oplus 1$).** $R_0$ sends its $\{S_0, S_1\}$ half to $R_1$; $R_1$ sends its $\{S_2, S_3\}$ half to $R_0$. Symmetric for $(R_2, R_3)$. Each rank sends $M/2$.

```
After AG step 2:

R0: [S0   S1   S2   S3]
R1: [S0   S1   S2   S3]
R2: [S0   S1   S2   S3]
R3: [S0   S1   S2   S3]
```

AR done in $2\lceil \log_2 4 \rceil = 4$ sequential steps.

**Cost accounting.** Per-step payloads follow a geometric schedule:

| Phase | Step | Payload | Per-step cost |
|---|---|---|---|
| RS | 1 | $M/2$ | $\alpha + (M/2)/\mathrm{BW}$ |
| RS | 2 | $M/4$ | $\alpha + (M/4)/\mathrm{BW}$ |
| AG | 1 | $M/4$ | $\alpha + (M/4)/\mathrm{BW}$ |
| AG | 2 | $M/2$ | $\alpha + (M/2)/\mathrm{BW}$ |

Total at $N=4$: $4\alpha + (M/2 + M/4 + M/4 + M/2)/\mathrm{BW} = 4\alpha + (3/2)\,M/\mathrm{BW}$. The bandwidth series is $M \cdot \sum_{k=1}^{\log_2 N} 2^{-k} = M \cdot (N-1)/N$ for one phase, doubled for the two-phase RS+AG. Generalizing:

$$t_{\mathrm{Rabenseifner\,AR}} = 2\lceil \log_2 N \rceil \, \alpha + 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

At $N=4$ this evaluates to $4\alpha + 1.5\,M/\mathrm{BW}$, matching the trace above.

**Strengths.** Bandwidth-optimal (same $2(N-1)/N$ BW floor as ring) at $O(\log N)$ latency instead of ring's $O(N)$. Strictly dominates both ring (same BW, fewer α) and simple recursive doubling (same α order, better BW) on α-β arithmetic. This is the "best on paper" AR algorithm for power-of-2 $N$.

**Weakness.** Same partner-cycling failure mode as simple recursive doubling — each RS step and each AG step exchanges with a different partner, so pipelining requires $\lceil \log_2 N \rceil$ concurrent ports per rank. On bounded-port fabrics (2 ring directions on NVLink, 2–8 NICs off-node, small-digit scale-up switch uplinks), the pipeline serializes and the non-pipelined BW term of $2(N-1)/N \cdot M/\mathrm{BW}$ is the best it attains. DBT (§3.2), which starts from a worse $\log_2 N \cdot M/\mathrm{BW}$ non-pipelined BW but **pipelines successfully** on a 3-port-budget fabric, matches or beats Rabenseifner's asymptote — which is why NCCL ships DBT and not Rabenseifner despite Rabenseifner's more attractive paper cost. Also strictly needs $N = 2^k$; non-power-of-2 requires a reduction-first embedding step that breaks the clean halving geometry.

---

## Appendix B: Non-mainline AG / RS / A2A variants

Three variants extend the main-text ring-based AG / RS (§4.1) and pairwise direct-send A2A (§5.1) with log-depth alternatives that each address a different tradeoff: B.1 (recursive-doubling AG / recursive-halving RS) is the MPI-menu log-latency variant that doesn't ship on NCCL for the same bounded-port pipelining reasons as Rabenseifner AR above; B.2 (Parallel Aggregated Trees, PAT) is the NCCL-shipped scale-out AG / RS that re-engineers the partner schedule so log-depth latency composes with a bounded intermediate buffer at 1 rank per node; B.3 (Bruck A2A) is the log-depth A2A that MPI ships for small-$M$ control messages but that NCCL does not select in its single-algorithm default. Each section gives the step-by-step trace and the cost formula cited by the main-text comparison.

### B.1 Recursive-doubling AG / recursive-halving RS

Recursive-doubling AG is exactly Phase 2 of Rabenseifner AR in [Appendix A.2](#a2-rabenseifner-halving-doubling-ar), run standalone: each rank starts with one $M/N$-byte chunk and ends holding the full concatenation. Recursive-halving RS is the dual (Phase 1 of Rabenseifner run standalone): each rank starts with the full $M$-byte vector and ends holding one $M/N$-byte reduced chunk. Both use the same hypercube partner schedule ($i \oplus 2^{k-1}$ at step $k$) with the payload doubling in AG and halving in RS across $\lceil \log_2 N \rceil$ steps.

**AG trace at $N = 4$.** Start: $R_i$ holds only its own chunk $c_i$ of size $M/4$.

**Step 1 ($k = 1$, partner $i \oplus 1$).** Pairs $(R_0, R_1)$ and $(R_2, R_3)$ each exchange their one chunk. After: each rank holds 2 chunks ($M/2$ total).

```
R0: [c0   c1    ?    ?]
R1: [c0   c1    ?    ?]
R2: [ ?    ?   c2   c3]
R3: [ ?    ?   c2   c3]
```

**Step 2 ($k = 2$, partner $i \oplus 2$).** Pairs $(R_0, R_2)$ and $(R_1, R_3)$ each exchange their 2-chunk block. After: every rank holds all 4 chunks.

```
R0: [c0   c1   c2   c3]
R1: [c0   c1   c2   c3]
R2: [c0   c1   c2   c3]
R3: [c0   c1   c2   c3]
```

AG done in $\lceil \log_2 4 \rceil = 2$ steps. RS is the mirror image: start from a full vector, run the same partner schedule in reverse order (step 1 partner $i \oplus 2$, step 2 partner $i \oplus 1$), and the payload **halves** each step so each rank ends owning one fully-reduced $M/N$ chunk.

**Cost.** The per-step payload follows a geometric schedule. For AG with $\lceil \log_2 N \rceil$ steps, step $k$ ships $2^{k-1} \cdot M/N$ bytes per rank per direction; total per-rank ship = $M/N \cdot \sum_{k=1}^{\log_2 N} 2^{k-1} = (N-1) \cdot M/N$ bytes — matching ring's BW lower bound. RS is symmetric. Per-step cost $\alpha + 2^{k-1} M / (N \, \mathrm{BW})$; summing:

$$t_{\mathrm{rec\,doub\,AG}} = t_{\mathrm{rec\,halv\,RS}} = \lceil \log_2 N \rceil \, \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

where $M$ is the per-rank final total volume (for AG) or initial total volume (for RS).

**Strengths.** BW-optimal (matches ring) at $O(\log N)$ latency instead of ring's $O(N)$. On paper, strictly dominates ring AG / RS on α-β arithmetic for all $(N, M)$.

**Weakness.** The same partner-cycling issue from Rabenseifner AR applies: each step targets a different partner, so pipelining fails on bounded-port fabrics and the non-pipelined log-depth BW term of $(N-1)/N \cdot M/\mathrm{BW}$ is the best this algorithm attains. On a 2-port-budget fabric (NVLink pair, 2-NIC scale-out), pipelining ring ($P = N$ segments on the 2 neighbor links) already saturates the asymptotic floor; switching to rec-doubling gains the $\log_2 N \to N-1$ α reduction but at the cost of a pipeline-unfriendly schedule that can't amortize that α into the BW path. Also, the geometric payload makes per-round chunk sizes irregular, which complicates overlap with compute. Needs $N = 2^k$. For all these reasons NCCL does not ship rec-doubling AG / rec-halving RS; MPI (MPICH, OpenMPI) ships it and dispatches by message-size threshold.

### B.2 Parallel Aggregated Trees (PAT) — scale-out AG / RS

PAT [NCCL-PAT] ships in NCCL starting in the 2.23 release specifically for **inter-node AG / RS at 1 rank per node**, and only for that regime — intra-node traffic continues to use ring even when PAT is enabled. It is the first NCCL-shipped collective algorithm whose designed-for operating point is the **scale-out fabric** (NICs, IB or RoCE), as distinguished from the scale-up NVLink / NVSwitch island that ring and DBT already cover well. Understanding why scale-out needs a different algorithm — and why PAT's specific structure fits that niche — is the point of this section; we give the schedule at $N = 8$ and the cost after the scale-out motivation.

**Why scale-out AG / RS needs a different algorithm.** At scale-out the communicator $N$ is the **node count**, not the intra-island GPU count. Large training and inference jobs run with $N$ in the hundreds to thousands of nodes. Two properties flip relative to the intra-NVLink regime:

- **$\alpha$ is much larger.** Scale-out NIC $\alpha$ is in the microseconds (InfiniBand EDR/HDR ≈ 1–2 μs, plus a kernel-launch and proxy-thread floor on NCCL's scale-out path of a few additional μs). Ring AG at $N = 512$ nodes is $(N-1)\alpha \approx 511 \times 2\,\mu$s $\approx 1$ ms of pure α — swamping most realistic per-rank $M/N$ payloads in bandwidth cost.
- **Port budget is 1–2.** Scale-out endpoints present 1–2 active NIC ports per rank (GPUDirect RDMA from one GPU typically drives one NIC, though 2–8 NICs per node exist). The partner-cycling schedule that kills rec-doubling on-node (needs $\log_2 N$ concurrent partners at steady state) kills it even harder at scale-out.

Ring's $(N-1)\alpha$ dominates; rec-doubling's pipeline fails; a different schedule is needed.

**PAT's key move — reversed Bruck + bounded buffer.** PAT uses the Bruck-style offset family ($i \pm 2^k$ for $k = 0, \ldots, \lceil \log_2 N \rceil - 1$) but runs the offsets **in reverse order**, farthest-first: at $N = 8$ the partner offset sequence is $4, 2, 1$ rather than Bruck's $1, 2, 4$. The message is split into $N$ chunks of $M/N$ bytes; at each round, rank $i$ exchanges **exactly one such chunk** (bounded-buffer) with its current partner, chosen so that after $\lceil \log_2 N \rceil$ rounds every rank holds the complete set. The AG is structured as $N$ parallel binary-tree broadcasts (one per source chunk), each tree reaching all $N$ ranks in $\log_2 N$ hops — hence "Parallel Aggregated Trees". Two properties fall out:

1. **Per-link payload per round is constant at $M/N$.** Unlike rec-doubling AG (payload $M/N, 2M/N, 4M/N, \ldots$), PAT does not let the exchange buffer grow. The intermediate buffer required per rank is therefore bounded by $O(M/N)$, which matters because scale-out proxy threads stage data through finite registered-memory buffers and don't have room for the full $M$.
2. **Partner order "farthest first" gets long-RTT transfers in flight early.** On scale-out, the longest-logical-distance exchange (offset $N/2$) also corresponds to the physically most distant node pair (worst-case cross-fabric RTT). Scheduling it first means the long transfer overlaps with the later, shorter-offset transfers on the NIC's DMA engines. Bruck's "nearest first" would leave the long transfer on the critical path at the end, after the NIC has already drained its pipeline.

**Schedule at $N = 8$.** Each rank starts holding its own chunk $c_i$ of size $M/N = M/8$.

```
Round 1 (offset 4, partner i XOR 4):
  Pairs: (R0,R4), (R1,R5), (R2,R6), (R3,R7)
  Each pair exchanges its one chunk.
  After: R0 has {c0,c4}; R1 has {c1,c5}; … ; R7 has {c3,c7}.  [2 chunks each]

Round 2 (offset 2, partner i XOR 2):
  Pairs: (R0,R2), (R1,R3), (R4,R6), (R5,R7)
  Each pair exchanges one selected chunk of the "current owned" set
  (the one the other side still needs from this pair's perspective).
  After: each rank holds 4 chunks of the 8. [4 chunks each]

Round 3 (offset 1, partner i XOR 1):
  Pairs: (R0,R1), (R2,R3), (R4,R5), (R6,R7)
  Each pair exchanges one selected chunk.
  After: every rank holds all 8 chunks. [8 chunks — complete]
```

The "one selected chunk" per round per pair is determined by the chunk-to-tree assignment: round $r$ advances, for every source $s$, exactly the one edge of $s$'s broadcast tree at depth $r$. With $N$ trees each of depth $\log_2 N$ running concurrently, the round advances all trees one level at once. The bounded-buffer property follows because a rank participates in exactly one tree-edge per round per direction.

**Cost accounting.** $\lceil \log_2 N \rceil$ sequential rounds, each shipping $M/N$ bytes per link, yielding a per-round cost of $\alpha + (M/N)/\mathrm{BW}$. Total per-rank on-wire volume is $(N-1)\cdot M/N$ — the AG BW lower bound — accumulated across the $\lceil \log_2 N \rceil$ rounds. Crucially the bandwidth term is **not** $\lceil \log_2 N \rceil \cdot M/N \cdot \mathrm{BW}^{-1}$ (which would be log-depth × per-round payload summed naïvely) because on a full-duplex link each rank both sends and receives each round, and across the $\lceil \log_2 N \rceil$ rounds the total data each rank actually ships out is exactly $(N-1) \cdot M/N$ bytes — one per chunk it doesn't originate, forwarded once on its outbound direction:

$$t_{\mathrm{PAT}} = \lceil \log_2 N \rceil \, \alpha + \frac{N-1}{N} \cdot \frac{M}{\mathrm{BW}}$$

RS is symmetric (reverse the schedule and replace forward-concat with sum-in).

**Why PAT ships only for inter-node, 1 rank per node.** Two constraints shape the shipping scope:

1. **Intra-node doesn't need PAT.** On a scale-up NVLink domain, ring AG's $(N-1)\alpha$ with $N \leq 72$ (NVL72) and $\alpha \approx 0.5\,\mu$s is $\sim 36\,\mu$s of latency — already small, and ring's pipelining keeps BW at the floor. Replacing it with PAT would trade pipeline-friendliness for a log-depth α schedule without any α budget to recover on the NVLink side. NCCL 2.23's implementation explicitly routes intra-node traffic through ring.
2. **Multiple ranks per node breaks the tree structure.** PAT's chunk-to-tree assignment assumes a flat $N$-rank communicator where each rank has a distinct, symmetric role in the trees. With $G > 1$ ranks per node, the intra-node ranks share a NIC (the scale-out port), and the tree-edge-per-round property degenerates: multiple tree-edges at the same round land on the same NIC and serialize, collapsing the log-depth advantage. The 2.23 release notes document this restriction directly; a future "multi-rank-per-node PAT" would need a hierarchical composition (intra-node ring / tree at the leaves, PAT between node leaders) that NCCL has not yet shipped.

The scope is therefore: PAT is the **inter-node scale-out AG / RS** algorithm at 1 rank per node; everywhere else NCCL stays on ring.

### B.3 Bruck A2A

Bruck's A2A [BHKUW97] is the canonical $O(\log N)$-latency A2A variant. It achieves $\lceil \log_2 N \rceil$ rounds — vs pairwise's $N-1$ — by running a butterfly / hypercube pattern on the bits of the destination index, combined with a pre-rotation and a post-rotation of the local buffer that together turn "send chunk to rank $j$" into "send the slots of my buffer whose current index has bit $k$ set to partner $i + 2^k$". The cost is a $\log_2 N / 2 \cdot M/\mathrm{BW}$ BW term — inflated relative to pairwise's $(N-1)/N \cdot M/\mathrm{BW}$ — plus $O(M)$ local rotation work that's not on the wire. We trace the schedule at $N = 4$ and give the cost.

**Pre-rotation.** Before any on-wire communication, rank $i$ cyclically rotates its send buffer **left by $i$ positions** (i.e., the chunk originally at slot $j$ moves to slot $(j - i) \bmod N$). The purpose is to align the "bit-$k$-of-destination-offset" pattern across ranks so that a single partner offset $+2^k$ per round suffices.

```
Initial (slots indexed by destination):       After pre-rotation (rank i shifts left by i):

R0: [v00  v01  v02  v03]                      R0: [v00  v01  v02  v03]   (i=0: no shift)
R1: [v10  v11  v12  v13]                      R1: [v11  v12  v13  v10]   (i=1: shift left by 1)
R2: [v20  v21  v22  v23]                      R2: [v22  v23  v20  v21]   (i=2: shift left by 2)
R3: [v30  v31  v32  v33]                      R3: [v33  v30  v31  v32]   (i=3: shift left by 3)
```

**Communication rounds ($\lceil \log_2 N \rceil = 2$ at $N=4$).** At round $k \in \{0, 1, \ldots, \lceil \log_2 N \rceil - 1\}$, rank $i$ sends to partner $(i + 2^k) \bmod N$ and receives from $(i - 2^k) \bmod N$. The chunks sent at round $k$ are the slots of the current buffer whose index has bit $k$ set ($N/2$ slots, total $M/2$ bytes per round); the chunks at the other $N/2$ slots stay put. The partner's corresponding slots overwrite the local ones on receive.

**Round 0 (partner $i+1 \bmod 4$, send slots with bit 0 set — i.e., slots 1 and 3, $M/2$ bytes).**

```
Sends (slots 1 and 3 of current buffer):
  R0 → R1: [v01, v03]          R1 → R2: [v12, v10]
  R2 → R3: [v23, v21]          R3 → R0: [v30, v32]

After round 0 (slots 1 and 3 overwritten by incoming):

R0: [v00  v30  v02  v32]        (slots 1,3 from R3)
R1: [v11  v01  v13  v03]        (slots 1,3 from R0)
R2: [v22  v12  v20  v10]        (slots 1,3 from R1)
R3: [v33  v23  v31  v21]        (slots 1,3 from R2)
```

**Round 1 (partner $i+2 \bmod 4$, send slots with bit 1 set — i.e., slots 2 and 3, $M/2$ bytes).**

```
Sends (slots 2 and 3 of current buffer):
  R0 → R2: [v02, v32]          R1 → R3: [v13, v03]
  R2 → R0: [v20, v10]          R3 → R1: [v31, v21]

After round 1 (slots 2 and 3 overwritten by incoming):

R0: [v00  v30  v20  v10]        (slots 2,3 from R2)
R1: [v11  v01  v31  v21]        (slots 2,3 from R3)
R2: [v22  v12  v02  v32]        (slots 2,3 from R0)
R3: [v33  v23  v13  v03]        (slots 2,3 from R1)
```

**Post-rotation.** Each rank $i$ cyclically rotates its buffer **right by $i$ positions** and optionally reverses the order (implementation detail; the net effect is to permute the received chunks into MPI / NCCL A2A output layout). After post-rotation:

```
Final (MPI/NCCL A2A output layout: slot j holds the chunk sent from R_j):

R0: [v00  v10  v20  v30]
R1: [v01  v11  v21  v31]
R2: [v02  v12  v22  v32]
R3: [v03  v13  v23  v33]
```

Matches the expected A2A end-state in §5.

**Cost.** The pre- and post-rotations are **local memory copies** (not on-wire; they do consume memory bandwidth and an intermediate buffer of size $M$, but in the α-β model we count on-wire time only). The $\lceil \log_2 N \rceil$ rounds each ship $M/2$ bytes per rank per direction, giving

$$t_{\mathrm{Bruck\,A2A}} = \lceil \log_2 N \rceil \, \alpha + \lceil \log_2 N \rceil \cdot \frac{M}{2\,\mathrm{BW}}$$

At $N = 4$: $2\alpha + M/\mathrm{BW}$, compared to pairwise's $3\alpha + (3/4)M/\mathrm{BW}$ at the same $N$. Bruck wins on α; pairwise wins on BW — the crossover is at $M^* \sim \alpha \cdot \mathrm{BW}$, computed in §5.2.

**Strengths.** Minimum-latency A2A (log-depth). The BW term is $\log_2 N / 2 \cdot M/\mathrm{BW}$ rather than pairwise's $(N-1)/N \cdot M/\mathrm{BW}$, so Bruck wins at small $M$ where the α term dominates.

**Weakness.** BW term scales with $\log_2 N$ rather than saturating at the lower bound; above the $M^*$ crossover, pairwise is strictly faster. The pre/post rotations require an intermediate buffer of $O(M)$ per rank, plus $O(M)$ local-memory copies that are free in the α-β model but non-trivial in practice. The bit-test routing is cleanest at $N = 2^k$; non-power-of-2 needs padding or a modified final round. Partner cycling across the $\log_2 N$ rounds is the same bounded-port objection that killed rec-doubling AR, though A2A is less bothered by it because there is no natural bandwidth-optimal log-depth A2A for pipelining to rescue — the inflated BW coefficient is fundamental, not a pipelining artifact. NCCL's single-algorithm policy picks pairwise for its large-$M$ optimality; MPI ships both Bruck and pairwise and dispatches by message-size threshold.

---

## Further reading

- **`02_topology_mapping.md`** — how ring / tree / log algorithms map to star (crossbar) and torus fabrics, with latency + BW derivations per topology, the torus dim-decomp AR mechanism worked out on a 2×2 example, and a side-by-side comparison at fixed $G$.
- **`03_in_network_collectives.md`** — how switch-resident reduction engines (SHARP, NVLS) replace $O(N)$ endpoint hops with $O(1)$ switch hops on star topologies.
- **`04_contention_and_congestion.md`** — how dynamic contention coefficients $\eta_\alpha \ge 1$ and $\eta_\beta \in (0, 1]$ modify the idealized costs above under realistic traffic.
- **`references.md`** — primary-source bibliography (Hockney's α-β model, Patarasuk-Yuan ring AR, Thakur-Rabenseifner-Gropp RHD, Sanders-Speck-Träff DBT, Bruck log A2A, PAT / NCCL 2.23 scale-out AG/RS, Jeaugey et al. "Demystifying NCCL" 2025).
- **Patarasuk & Yuan (2009)**, "Bandwidth optimal all-reduce algorithms for clusters of workstations" — the foundational proof that ring AR achieves the $2(N-1)/N \cdot M/\mathrm{BW}$ bandwidth lower bound.
