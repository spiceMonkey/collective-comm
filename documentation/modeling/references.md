# References: Collectives Explainer Series

**Author:** Yue Lu  
**Date:** April 2026  

Self-contained bibliography for `01–05` in this folder. Every non-obvious equation and empirical value in the series has been cross-checked against the sources below.

---

## Primary algorithm papers

**[ALPHA-BETA]** Hockney, R. (1994). *The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.* Parallel Computing 20(3). → Canonical α-β latency model $t = \alpha + M/\mathrm{BW}$. Used throughout `01_collective_algorithms.md` §1.

**[PY09]** Patarasuk, P., & Yuan, X. (2009). *Bandwidth Optimal All-Reduce Algorithms for Clusters of Workstations.* JPDC 69(2):117–124. → Ring AR achieves $2(N-1)/N \cdot M/\mathrm{BW}$ on any tree-connected fabric. Backs up `01_collective_algorithms.md` §5.1 and `02_topology_mapping.md` §3.4. <https://doi.org/10.1016/j.jpdc.2008.09.002>

**[TRG05]** Thakur, R., Rabenseifner, R., & Gropp, W. (2005). *Optimization of Collective Communication Operations in MPICH.* IJHPCA 19(1):49–66. → Recursive halving-doubling all-reduce achieves $2\lceil \log_2 N \rceil \alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ for power-of-2 N; describes the ring / RHD / Rabenseifner crossover rules implemented by modern MPI libraries. Backs up `01_collective_algorithms.md` Appendix B.2, `01_collective_algorithms.md` Appendix B.4, and `02_topology_mapping.md` Appendix A. <https://journals.sagepub.com/doi/10.1177/1094342005051521>, author preprint: <https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf>

**[CHPV07]** Chan, E., Heimlich, M., Purkayastha, A., & van de Geijn, R. (2007). *Collective Communication: Theory, Practice, and Experience.* Concurrency and Computation: Practice and Experience, 19(13):1749–1783. → Dim-decomposed all-reduce framework and telescoping derivation of multi-dim ring costs. Backs up the torus dim-decomp cost and BW telescoping in `02_topology_mapping.md` §3.1.

**[BHKUW97]** Bruck, J., Ho, C.-T., Kipnis, S., Upfal, E., & Weathersby, D. (1997). *Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems.* IEEE TPDS 8(11):1143–1156. → Log-round all-to-all via circular shifts and bit-wise exchange; $\lceil \log_2 N \rceil$ steps at the cost of moving the full payload per step. Backs up the "log A2A" variant in `01_collective_algorithms.md` Appendix B.5. <https://doi.org/10.1109/71.642949>

**[SST09]** Sanders, P., Speck, J., & Träff, J.L. (2009). *Two-Tree Algorithms for Full Bandwidth Broadcast, Reduction and Scan.* Parallel Computing 35(12):581–594. → Double binary tree (DBT) construction: two complementary trees that each rank is interior in exactly one of, used to saturate both directions of every full-duplex link during broadcast / reduce / AR. Backs up the DBT derivation in `01_collective_algorithms.md` §5.2, the DBT shipping rationale in `01_collective_algorithms.md` §5.3, and the binomial-tree BC / Reduce cost in `01_collective_algorithms.md` §3.2 and §4.2. <https://doi.org/10.1016/j.parco.2009.09.001>

**[LEIS85]** Leiserson, C.E. (1985). *Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing.* IEEE Transactions on Computers C-34(10):892–901. → Foundational fat-tree network construction: universal routing with bandwidth that grows fatter toward the root; the basis for Clos topology analysis in modern AI fabrics. Backs up the fat-tree / Clos topology treatment in `03_hierarchical_topologies.md` §1 and the Leiserson-vs-Al-Fares comparison in `03_hierarchical_topologies.md` Appendix B. <https://doi.org/10.1109/TC.1985.6312192>

**[AL-FARES08]** Al-Fares, M., Loukissas, A., & Vahdat, A. (2008). *A Scalable, Commodity Data Center Network Architecture.* ACM SIGCOMM 2008, pp. 63–74. → The k-ary fat-tree construction from commodity k-port switches: each switch splits its k ports k/2 up / k/2 down, yielding a rearrangeably non-blocking Clos with $s = 1$ at every tier. The equivalence "k-ary fat-tree = Clos with $s = 1$" that modern datacenter fabrics rely on is proved constructively in §3. Backs up the Clos-vs-fat-tree distinction in `03_hierarchical_topologies.md` §1 and the k-ary fat-tree derivation in `03_hierarchical_topologies.md` Appendix B. <https://doi.org/10.1145/1402958.1402967>

**[THAKUR-HIER]** Thakur, R., & Gropp, W. (2003). *Improving the Performance of Collective Operations in MPICH.* EuroPVM/MPI 2003, LNCS 2840:257–267. → Message-size-tuned algorithm selection for MPICH collectives (allgather, broadcast, reduce-scatter, reduce) — the conference precursor of [TRG05] — benchmarked on a Myrinet Linux cluster and an IBM SP. Background for the MPI algorithm menu in `01_collective_algorithms.md` Appendix B. It is not a hierarchical-scheduling paper; for the tier-aware pattern of `03_hierarchical_topologies.md` §2.1 see [MAGPIE]. <https://doi.org/10.1007/978-3-540-39924-7_38>

**[MAGPIE]** Kielmann, T., Hofman, R.F.H., Bal, H.E., Plaat, A., & Bhoedjang, R.A.F. (1999). *MagPIe: MPI's Collective Communication Operations for Clustered Wide Area Systems.* PPoPP 1999, pp. 131–140. → Hierarchical collective algorithms for wide-area clusters with heterogeneous tier bandwidths; introduces the principle of "do most of the work on the fastest tier" that informs the inner-vs-outer tier tradeoff in `03_hierarchical_topologies.md` §2. <https://doi.org/10.1145/301104.301116>

**[NCCL-PAT]** NVIDIA Corporation. (2024). *NCCL 2.23 Release Notes* (adds the PAT algorithm for AllGather and ReduceScatter). Accompanying blog: "New Scaling Algorithm and Initialization with NVIDIA Collective Communications Library 2.23," NVIDIA Developer Blog. Algorithm paper: Jeaugey, S. (2025). *PAT: a new algorithm for all-gather and reduce-scatter operations at scale.* arXiv:2506.20252. → PAT (Parallel Aggregated Trees): starts from the reversed-dimension Bruck schedule (far dimension first for all-gather); a logarithmic number of network transfers for small operations, with the payload aggregating per step; a logarithmic amount of internal buffering, independent of operation size; degrades toward a linear number of full-buffer transfers for large operations. Shipping scope: inter-node AG / RS only, at 1 rank per node. Backs up `01_collective_algorithms.md` §6 (scale-out rationale) and `01_collective_algorithms.md` Appendix A (full derivation). <https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-23-4.html>, <https://arxiv.org/abs/2506.20252>

**[DEMYST-NCCL]** Hu, Z., Shen, S., Bonato, T., Jeaugey, S., et al. (2025). *Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms.* arXiv:2507.04786. → Empirical tuner-trace analysis of NCCL's algorithm selection across AR / AG / RS at representative scales; confirms the Tree-vs-Ring selection rule (DBT for small-$M$, ring for large-$M$) on NVLink / NVSwitch fabrics and documents the per-regime crossover points that the α-β model alone does not predict. Backs up the "practice caveat" at the end of `01_collective_algorithms.md` §5.3 and the corresponding note in `02_topology_mapping.md` §2. <https://arxiv.org/abs/2507.04786>

---

## Topology architecture papers

**[TPU-V4]** Jouppi, N.P., et al. (2023). *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.* ISCA 2023. → 3D torus with optical circuit switching for slice reconfiguration; twisted-torus 1.63× A2A gain on asymmetric layouts (§V). Backs up the TPU torus deployment note in `02_topology_mapping.md` §3.4 and the OCS slice-reshaping commentary in `02_topology_mapping.md` §3.6; source for the torus realistic $\eta_\beta \approx 0.60$ calibration in `05_contention_and_congestion.md` §4.1. <https://doi.org/10.1145/3579371.3589350>

**[TRN2-ARCH]** Amazon Web Services. (2024–2025). *Amazon EC2 Trn2 Architecture.* AWS Neuron Documentation. → Trn2 server: 16 Trainium2 chips arranged as a 2D NeuronLink torus (each chip connects to 4 neighbors). Trn2 UltraServer: four Trn2 instances joined via a Z-dimension NeuronLink into a 3D 64-chip torus. Backs up the Trainium adoption note in `02_topology_mapping.md` §3.4 and §3.6. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html>

**[NEURON-CC]** Amazon Web Services. (2024–2025). *Neuron Collective Communication / Intra-node Collectives.* AWS Neuron Documentation. → NeuronX Collective Communication Library ships ring / mesh / KangaRing / RDH all-reduce variants purpose-built for Trainium's 2D/3D NeuronLink torus; per-NeuronCore CC Cores offload the orchestration of collective phases. Backs up the NeuronX CCL references in `02_topology_mapping.md` §3.4, §5.1 and Appendix A. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/intranode-collective-comm.html>

**[DGX-SUPERPOD]** NVIDIA Corporation. (2023–2024). *NVIDIA DGX SuperPOD Reference Architecture: Featuring NVIDIA DGX H100 / H200 / GB200 Systems.* NVIDIA Enterprise Documentation. → Canonical production reference architecture for InfiniBand-based AI training pods: full-bisection (s = 1) leaf-spine Clos within each Scalable Unit (SU), optional oversubscription between SUs at the super-spine when cost dominates over locality. Backs up the production-Clos deployment claim in `03_hierarchical_topologies.md` §1 and the rail-optimized layout discussion in `03_hierarchical_topologies.md` §1.1. <https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/>

**[META-RSC]** Meta AI. (2022). *Introducing the AI Research SuperCluster — Meta's cutting-edge AI supercomputer for AI research.* Meta AI Blog, Jan 2022. → Meta Research SuperCluster (RSC): 760 DGX A100 systems (6,080 A100 GPUs) at launch, planned to grow to 16,000 GPUs, on a 200 Gb/s InfiniBand two-level Clos fabric with no oversubscription; representative of production hyperscaler AI training deployments. Backs up the production-Clos deployment claim in `03_hierarchical_topologies.md` §1. <https://ai.meta.com/blog/ai-rsc/>

---

## In-network collectives

**[SHARP-IB]** Graham, R.L., et al. (2016). *Scalable Hierarchical Aggregation Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction.* COMHPC Workshop at SC16. → Original SHARP specification for InfiniBand; hardware-offloaded reduction trees; reports ~95% of network bandwidth utilization and 2–5× speedup over host-based reduction. Backs up the in-network reduction (switch ALU) mechanism in `04_in_network_collectives.md` §1.1 and the SHARP-class Quantum SHARP / Spectrum-X SHARP commercial discussion in `04_in_network_collectives.md` §2.2. <https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf>

**[NVLINK-SHARP]** NVIDIA Corporation. (2023–2024). *NVLink SHARP (NVLS) — In-Network All-Reduce Acceleration.* GTC S62129; NCCL release notes; GB200 NVL Multi-Node Tuning Guide. → NVSwitch-based in-network reduction; AR busbw rises from ~360 GB/s to 470+ GB/s under NVLS (≈1.3× measured). Backs up the NVLS mechanism in `04_in_network_collectives.md` §1.1 and the scale-up INC commercial summary in `04_in_network_collectives.md` §2.1. The measured 1.3× AR BW lift is the calibration source for the `03_hierarchical_topologies.md` §3.1 inner-tier-INC commentary and for the NVLS-row $\eta_\beta \approx 0.52$ in `05_contention_and_congestion.md` §4.1. <https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html>

**[NCCL-NVLS-2.27]** NVIDIA Corporation. (2025). *Enabling Fast Inference and Resilient Training with NCCL 2.27.* NVIDIA Technical Blog, May 2025. → Up to 2.5× improvement on small-to-medium messages in Symmetric Memory configurations with NVLS. Backs up the small-$M$ speedup regime discussion in `04_in_network_collectives.md` §3.2. <https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/>

**[TH-ULTRA]** Broadcom. (2025). *Broadcom Ships Tomahawk Ultra: Reimagining the Ethernet Switch for HPC and AI Scale-up.* Jul 2025. → First Ethernet switch ASIC with in-network collectives (INC); 51.2 Tbps full-duplex, 250 ns switch latency. Backs up the Tomahawk Ultra HW A2A mention in `04_in_network_collectives.md` §1.3 and the Ethernet INC entry in `04_in_network_collectives.md` §2.1. <https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-tomahawk-ultra-reimagining-ethernet-switch-hpc>

---

## Hardware specifications

**[NVLINK5-SPEC]** NVIDIA Corporation. (2024). *NVLink 5 / NVSwitch Gen4 / GB200 NVL72 Architecture.* NVIDIA product pages. → NVLink 5: 18 links × 100 GB/s bidirectional (50 GB/s per direction) = **1.8 TB/s bidirectional per GPU** (900 GB/s unidirectional). NVSwitch Gen4: 72 NVLink 5 ports per chip. NVL72: 72 GPUs, 130 TB/s aggregate all-to-all bandwidth. Backs up the NVLink-5 bandwidth numbers in `02_topology_mapping.md` §2 and `03_hierarchical_topologies.md` §1.1–§1.2. <https://www.nvidia.com/en-us/data-center/nvlink/>

**[UCIE]** UCIe Consortium. (2023–2024). *Universal Chiplet Interconnect Express (UCIe) Specification v1.1 / v2.0.* → Standard die-to-die interface for chiplet interconnects; per-lane signaling rates up to 32 GT/s (v1.1 and v2.0; 48 / 64 GT/s arrive with v3.0 in 2025), target latency ~2 ns per hop for advanced package PHY. Backs up the chiplet-mesh $\alpha$ and BW numbers in `02_topology_mapping.md` §1.3 and §4.1. <https://www.uciexpress.org/specifications>

**[NCCL-TESTS]** NVIDIA Corporation. (2020–2025). *NCCL Performance Tests.* → Canonical busbw/algbw measurement methodology for NCCL collectives; H100 intra-node AR busbw ≈ 360 GB/s against 450 GB/s peak NVLink4 unidirectional. Backs up the busbw/algbw vocabulary in `01_collective_algorithms.md` Appendix D and is the calibration source for the crossbar $\eta_\beta \approx 0.80$ in `05_contention_and_congestion.md` §4.1. <https://github.com/NVIDIA/nccl-tests>

**[H100-SPEC]** NVIDIA Corporation. (2022). *NVIDIA H100 Tensor Core GPU Architecture.* NVIDIA Whitepaper WP-10792-001. → H100 SXM5 NVLink 4.0: 900 GB/s bidirectional per GPU (450 GB/s unidirectional). Baseline "pre-NVLink5" BW numbers behind the calibration in `05_contention_and_congestion.md` §2.1 and §4.1. <https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet>

**[QX800]** NVIDIA Corporation. (2024). *NVIDIA Quantum-X800 InfiniBand Platform / Q3400 Switch Series.* NVIDIA product pages and datasheet. → 144-port 800 Gbps (XDR) InfiniBand switch (air-cooled Q3400-RA plus a liquid-cooled variant); 115.2 Tbps aggregate bidirectional throughput; designed for GB200 NVL72 SuperPOD scale-out fabric. Backs up the 144-port ToR radix used in both pod-local and rail-optimized X800 layouts in `03_hierarchical_topologies.md` §1.1. <https://www.nvidia.com/en-us/networking/quantum-x800/>

---

## Tag → section map

Where each source is cited or relied on in the series:

| Tag | Used in |
|---|---|
| `[ALPHA-BETA]` | `01_collective_algorithms.md` §1 |
| `[PY09]` | `01_collective_algorithms.md` §5.1, `02_topology_mapping.md` §3.4 |
| `[TRG05]` | `01_collective_algorithms.md` App. B.2, App. B.4; `02_topology_mapping.md` App. A |
| `[SST09]` | `01_collective_algorithms.md` §3.2, §4.2, §5.2, App. C.5; `02_topology_mapping.md` §5.1; `00_summary.md` §4 |
| `[CHPV07]` | `02_topology_mapping.md` §3.1 |
| `[BHKUW97]` | `01_collective_algorithms.md` App. B.5 |
| `[NCCL-PAT]` | `01_collective_algorithms.md` §6, App. A |
| `[DEMYST-NCCL]` | `01_collective_algorithms.md` §5.3; `02_topology_mapping.md` §2 |
| `[LEIS85]` | `03_hierarchical_topologies.md` §1, App. B (B.1) |
| `[AL-FARES08]` | `03_hierarchical_topologies.md` §1, App. B (B.2) |
| `[THAKUR-HIER]` | `01_collective_algorithms.md` App. B (background) |
| `[MAGPIE]` | `03_hierarchical_topologies.md` §2 (inner-vs-outer tier tradeoff) |
| `[TPU-V4]` | `02_topology_mapping.md` §3.4, §3.6; `05_contention_and_congestion.md` §4.1 |
| `[TRN2-ARCH]` | `02_topology_mapping.md` §3.4, §3.6 |
| `[NEURON-CC]` | `02_topology_mapping.md` §3.4, §5.1, App. A |
| `[DGX-SUPERPOD]` | `03_hierarchical_topologies.md` §1, §1.1 |
| `[META-RSC]` | `03_hierarchical_topologies.md` §1 |
| `[NCCL-TESTS]` | `01_collective_algorithms.md` App. D; `05_contention_and_congestion.md` §4.1 |
| `[SHARP-IB]` | `04_in_network_collectives.md` §1.1, §2.2 |
| `[NVLINK-SHARP]` | `04_in_network_collectives.md` §1.1, §2.1; `03_hierarchical_topologies.md` §3.1; `05_contention_and_congestion.md` §4.1 |
| `[NCCL-NVLS-2.27]` | `04_in_network_collectives.md` §3.2 |
| `[TH-ULTRA]` | `04_in_network_collectives.md` §1.3, §2.1 |
| `[NVLINK5-SPEC]` | `02_topology_mapping.md` §2; `03_hierarchical_topologies.md` §1.1, §1.2 |
| `[UCIE]` | `02_topology_mapping.md` §1.3, §4.1 |
| `[H100-SPEC]` | `05_contention_and_congestion.md` §2.1, §4.1 |
| `[QX800]` | `03_hierarchical_topologies.md` §1.1 |

---

## Verification notes (what was cross-checked)

The following equations and empirical values were independently verified (not just transcribed from secondary sources):

| Claim | Verified against | Status |
|---|---|---|
| Ring AR cost $2(N-1)\alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ | `[PY09]` direct formula | ✓ |
| Tree (RHD) AR $2 \lceil \log_2 N \rceil \alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ | `[TRG05]` §3.3 | ✓ |
| Torus dim-decomp $2 \sum(d_i-1)\alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ | `[CHPV07]`, `[PY09]` telescoping | ✓ |
| NVL72 = 72 GPUs per NVSwitch domain | `[NVLINK5-SPEC]` | ✓ |
| NVLink 5 per-GPU unidirectional 900 GB/s | `[NVLINK5-SPEC]`: 18 × 50 GB/s per direction | ✓ |
| Log A2A $\lceil \log_2 N \rceil$ steps | `[BHKUW97]` §III | ✓ |
| NVLS measured 470+ GB/s busbw | `[NVLINK-SHARP]` GB200 tuning guide | ✓ (≈1.3× over 360 GB/s non-SHARP; paper's "1.7×" is the **IB Quantum-2** figure, not NVLS) |
| Tomahawk Ultra 250 ns switch latency | `[TH-ULTRA]` | ✓ |
| Crossbar $\eta_\beta \approx 0.80$ (360/450) | `[NCCL-TESTS]` PERFORMANCE.md | ✓ |
| Torus $\eta_\beta \approx 0.60$ from twisted-torus 1.63× | `[TPU-V4]` §V | ✓ (upper-bound interpretation) |

### Known numerical caveats (documented in-line)

- The $\alpha$ values used in worked examples ($\alpha = 0.5\\,\mu$s intra-NVLink; $\alpha \approx 2\\,\mu$s / $8\\,\mu$s for the same-ToR / via-spine InfiniBand paths of `03_hierarchical_topologies.md` §1.2) are order-of-magnitude estimates; 0.5 μs corresponds to NVSwitch cut-through (~100 ns) plus endpoint software (~400 ns). Exact values depend on the NCCL path chosen at runtime. The model's conclusions are robust to $\pm 2\times$ variation on $\alpha$ at bulk payloads, where the bandwidth term dominates — above roughly 1 MB for INC AR and 4 MB for DBT AR at $N = 512$ (`04_in_network_collectives.md` §3.2); ring AR at $N = 512$ stays α-dominated up to about 230 MB.
- The A2A on star entries in `02_topology_mapping.md` §2.5 and §5.1 use the per-rank pairwise direct-send formula (NVSwitch aggregate bisection saturated), appropriate for MoE workloads where all ranks participate simultaneously.
