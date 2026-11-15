# References: Collectives Explainer Series

**Author:** Yue Lu  
**Date:** April 2026  

Self-contained bibliography for `01–05` in this folder. Every non-obvious equation and empirical value in the series has been cross-checked against the sources below; the verification notes at the end say which values are vendor-published and which are community-reported.

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

**[THAKUR-HIER]** Thakur, R., & Gropp, W. (2003). *Improving the Performance of Collective Operations in MPICH.* EuroPVM/MPI 2003, LNCS 2840:257–267. → Message-size-tuned algorithm selection for MPICH collectives (allgather, broadcast, reduce-scatter, reduce) — the conference precursor of [TRG05] — benchmarked on a Myrinet Linux cluster and an IBM SP. Background for the MPI algorithm menu in `01_collective_algorithms.md` Appendix B. It is not a hierarchical-scheduling paper; for the tier-aware pattern of `03_hierarchical_topologies.md` §2.1 see [MAGPIE], [ZGGT09] and [HAN]. <https://doi.org/10.1007/978-3-540-39924-7_38>

**[MAGPIE]** Kielmann, T., Hofman, R.F.H., Bal, H.E., Plaat, A., & Bhoedjang, R.A.F. (1999). *MagPIe: MPI's Collective Communication Operations for Clustered Wide Area Systems.* PPoPP 1999, pp. 131–140. → Hierarchical collective algorithms for wide-area clusters with heterogeneous tier bandwidths; introduces the principle of "do most of the work on the fastest tier" that informs the inner-vs-outer tier tradeoff in `03_hierarchical_topologies.md` §2. <https://doi.org/10.1145/301104.301116>

**[ZGGT09]** Zhu, H., Goodell, D., Gropp, W., & Thakur, R. (2009). *Hierarchical Collectives in MPICH2.* EuroPVM/MPI 2009, LNCS 5759. → An implementation of the MPI collectives on message-passing primitives that exploits the two-level hierarchy of multi-core clusters (within a node, between nodes). Backs up the tier-aware framing of `03_hierarchical_topologies.md` §2. <https://doi.org/10.1007/978-3-642-03770-2_41>

**[HAN]** Luo, X., Wu, W., Bosilca, G., Pei, Y., Cao, Q., Patinyasakdikul, T., Zhong, D., & Dongarra, J. (2020). *HAN: a Hierarchical AutotuNed Collective Communication Framework.* IEEE CLUSTER 2020. → Open MPI's hierarchical collective framework: a collective is split by hardware level (intra-node, inter-node), each level running its own tuned algorithm. Backs up the "different primitive per tier" framing of `03_hierarchical_topologies.md` §2. <https://ieeexplore.ieee.org/document/9229573>

**[NCCL-PAT]** NVIDIA Corporation. (2024). *NCCL 2.23 Release Notes* (adds the PAT algorithm for AllGather and ReduceScatter). Accompanying blog: "New Scaling Algorithm and Initialization with NVIDIA Collective Communications Library 2.23," NVIDIA Developer Blog. Algorithm paper: Jeaugey, S. (2025). *PAT: a new algorithm for all-gather and reduce-scatter operations at scale.* arXiv:2506.20252. → PAT (Parallel Aggregated Trees): starts from the reversed-dimension Bruck schedule (far dimension first for all-gather); a logarithmic number of network transfers for small operations, with the payload aggregating per step; a logarithmic amount of internal buffering, independent of operation size; degrades toward a linear number of full-buffer transfers for large operations. Shipping scope: inter-node AG / RS only, at 1 rank per node. Backs up `01_collective_algorithms.md` §6 (scale-out rationale) and `01_collective_algorithms.md` Appendix A (full derivation). <https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-23-4.html>, <https://arxiv.org/abs/2506.20252>

**[DEMYST-NCCL]** Hu, Z., Shen, S., Bonato, T., Jeaugey, S., et al. (2025). *Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms.* arXiv:2507.04786. → Empirical tuner-trace analysis of NCCL's algorithm selection across AR / AG / RS at representative scales; confirms the Tree-vs-Ring selection rule (DBT for small-$M$, ring for large-$M$) on NVLink / NVSwitch fabrics and documents the per-regime crossover points that the α-β model alone does not predict. Backs up the "practice caveat" at the end of `01_collective_algorithms.md` §5.3 and the corresponding note in `02_topology_mapping.md` §2. <https://arxiv.org/abs/2507.04786>

---

## Topology architecture papers

**[TPU-V4]** Jouppi, N.P., et al. (2023). *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.* ISCA 2023. → 3D torus with optical circuit switching for slice reconfiguration; twisted-torus 1.63× A2A gain on asymmetric layouts (§V). Backs up the TPU torus deployment note in `02_topology_mapping.md` §3.4 and the OCS slice-reshaping commentary in `02_topology_mapping.md` §3.6; source for the torus realistic $\eta_\beta \approx 0.60$ calibration in `05_contention_and_congestion.md` §4.1. <https://doi.org/10.1145/3579371.3589350>

**[TRN2-ARCH]** Amazon Web Services. (2024–2025). *Amazon EC2 Trn2 Architecture.* AWS Neuron Documentation. → Trn2 server: 16 Trainium2 chips arranged as a 2D NeuronLink torus (each chip connects to 4 neighbors). Trn2 UltraServer: four Trn2 instances joined via a Z-dimension NeuronLink into a 3D 64-chip torus. Backs up the Trainium adoption note in `02_topology_mapping.md` §3.4 and §3.6. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html>

**[NEURON-CC]** Amazon Web Services. (2024–2025). *Neuron Collective Communication / Intra-node Collectives.* AWS Neuron Documentation. → NeuronX Collective Communication Library ships ring / mesh / KangaRing / RDH all-reduce variants purpose-built for Trainium's 2D/3D NeuronLink torus; per-NeuronCore CC Cores offload the orchestration of collective phases. Backs up the NeuronX CCL references in `02_topology_mapping.md` §3.4, §5.1 and Appendix A. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/intranode-collective-comm.html>

**[DGX-SUPERPOD]** NVIDIA Corporation. (2023–2025). *NVIDIA DGX SuperPOD Reference Architecture* (the DGX H100 edition is linked; the GB200 and GB300 editions are [DGX-SUPERPOD-GB200] and [DGX-SUPERPOD-GB300]). → Reference architecture family for InfiniBand-based AI training clusters: a non-blocking ($s = 1$), rail-optimized fat-tree compute fabric built from Scalable Units (SUs). Backs up the production-Clos deployment claim in `03_hierarchical_topologies.md` §1. <https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/>

**[DGX-SUPERPOD-GB200]** NVIDIA Corporation. (2024–2025). *NVIDIA DGX SuperPOD Reference Architecture Featuring NVIDIA DGX GB200 Systems* — Network Fabrics. → Compute fabric of 64-port Quantum-2 QM9700 switches at 400 Gb/s (NDR): four ConnectX-7 per compute tray, 18 trays and 72 GPUs per rack, 4 rails; one SU = 8 racks = 576 GPUs with one spine-leaf group per rail (8 leaf + 6 spine each, one leaf per rack); rail-optimized, non-blocking fat-tree; a core tier is added from 2 SUs (4 SUs / 2304 GPUs: 128 leaf, 96 spine, 36 core). Backs up the GB200-generation comparisons in `03_hierarchical_topologies.md` §1.1, §1.2 and Appendix A.4. <https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html>

**[DGX-SUPERPOD-GB300]** NVIDIA Corporation. (2025). *NVIDIA DGX SuperPOD: Next Generation Scalable Infrastructure for AI Factories — Reference Architecture Featuring NVIDIA DGX GB300 Systems.* → Compute fabric of 144-port Quantum-X800 Q3400 switches at 800 Gb/s (XDR): four ConnectX-8 per compute tray; rail-optimized, non-blocking, full fat-tree; one SU = 8 racks = 576 GPUs with eight leaf switches, two per rail; a two-layer fabric up to 16 SUs (9,216 GPUs); Table 3 gives 32 leaf + 18 spine switches for 4 SUs (2304 GPUs); Figure 8 draws every spine linked to the leaves of all four rails. Backs up the rail-optimized layout of `03_hierarchical_topologies.md` §1.1, the outer-tier bandwidth of §1.2 and the switch counts of Appendix A. <https://docs.nvidia.com/pdf/dgx-spod-gb300-ra.pdf>

**[NCCL-PXN]** NVIDIA Corporation. (2022). *Doubling all2all Performance with NVIDIA Collective Communication Library 2.12.* NVIDIA Technical Blog. → PXN (PCI × NVLink): a GPU can send through a NIC that is not its own by first moving the data over NVLink to the GPU attached to that NIC; on rail-optimized fabrics (NIC $i$ of every node on leaf $i$) this keeps messages on one rail's leaf switch and reduces traffic through the spine tier. Backs up the NVLink rail bridge in `03_hierarchical_topologies.md` §1.1 and §2.2. <https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/>

**[META-RSC]** Meta AI. (2022). *Introducing the AI Research SuperCluster — Meta's cutting-edge AI supercomputer for AI research.* Meta AI Blog, Jan 2022. → Meta Research SuperCluster (RSC): 760 DGX A100 systems (6,080 A100 GPUs) at launch, planned to grow to 16,000 GPUs, on a 200 Gb/s InfiniBand two-level Clos fabric with no oversubscription; representative of production hyperscaler AI training deployments. Backs up the production-Clos deployment claim in `03_hierarchical_topologies.md` §1. <https://ai.meta.com/blog/ai-rsc/>

---

## In-network collectives

**[SHARP-IB]** Graham, R.L., et al. (2016). *Scalable Hierarchical Aggregation Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction.* COMHPC Workshop at SC16. → Original SHARP specification for InfiniBand; hardware-offloaded reduction trees; reports ~95% of network bandwidth utilization and 2–5× speedup over host-based reduction. Backs up the in-network reduction (switch ALU) mechanism in `04_in_network_collectives.md` §1.1 and the Quantum SHARP commercial discussion in `04_in_network_collectives.md` §2.2. <https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf>

**[NVLINK-SHARP]** NVIDIA Corporation. (2023–2025). *NVLink SHARP (NVLS).* NCCL 2.17.1 release notes (NVLink SHARP reduction / broadcast support for intra-node all-reduce); GB200 NVL Multi-Node Tuning Guide, NCCL chapter (NVLS is enabled by default where supported; `NCCL_NVLS_ENABLE`). → NVSwitch-based in-network reduction and multicast. Backs up the NVLS mechanism in `04_in_network_collectives.md` §1.1 and the scale-up INC summary in `04_in_network_collectives.md` §2.1. Neither source publishes busbw numbers; the measured lift quoted in this series comes from [NCCL-NVLS-REPORTS]. <https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-17-1.html>, <https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html>

**[NCCL-NVLS-REPORTS]** Community reports in NVIDIA's public issue trackers (not vendor-published benchmarks). `NVIDIA/nccl-tests` issue #312 (May 2025): an 8-GPU all-reduce run with NVLS enabled reports busbw of 470–482 GB/s at 1–8 GB messages; the reporter does not name the GPU model, gives 450 GB/s as the theoretical figure for H20 / H100 / H200, and states (without measuring it in that run) a typical practical maximum of about 360 GB/s. `NVIDIA/nccl` issue #1221 (March 2024): on 8×H100 nodes, enabling NVLS moved bus bandwidth from about 330 to about 430 GB/s on one or two nodes (no gain beyond two). → The ~1.3× NVLS lift and the 360 / 450 ratio behind the crossbar $\eta_\beta \approx 0.80$ and the NVLS $\eta_\beta \approx 0.52$ in `05_contention_and_congestion.md` §4.1, and the lift quoted in `03_hierarchical_topologies.md` §3 and `04_in_network_collectives.md` §1.4 and §4; indicative values only. <https://github.com/NVIDIA/nccl-tests/issues/312>, <https://github.com/NVIDIA/nccl/issues/1221>

**[NCCL-NVLS-2.27]** NVIDIA Corporation. (2025). *Enabling Fast Inference and Resilient Training with NCCL 2.27.* NVIDIA Technical Blog, Jul 2025. → NCCL 2.27 brings SHARP support to AllGather and ReduceScatter when NVLink SHARP is combined with IB SHARP; its symmetric-memory kernels give up to 2.5× on small-to-medium messages. Backs up the Quantum SHARP entry in `04_in_network_collectives.md` §2.2. <https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/>

**[TH-ULTRA]** Broadcom. (2025). *Broadcom Ships Tomahawk Ultra: Reimagining the Ethernet Switch for HPC and AI Scale-up.* Press release, 15 Jul 2025. → Ethernet switch ASIC with in-network collectives — the release names AllReduce, Broadcast and AllGather as operations executed on the chip, and no all-to-all; 51.2 Tbps, 250 ns switch latency. Backs up the Tomahawk Ultra entries in `04_in_network_collectives.md` §1.1, §1.2 and §2.1, and the statement in §1.3 that it lists no all-to-all primitive. <https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-tomahawk-ultra-reimagining-ethernet-switch-hpc>

---

## Hardware specifications

**[NVLINK5-SPEC]** NVIDIA Corporation. (2024). *NVLink 5 / NVSwitch Gen4 / GB200 NVL72 Architecture.* NVIDIA product pages. → NVLink 5: 18 links × 100 GB/s bidirectional (50 GB/s per direction) = **1.8 TB/s bidirectional per GPU** (900 GB/s unidirectional). NVSwitch Gen4: 72 NVLink 5 ports per chip. NVL72: 72 GPUs, 130 TB/s aggregate all-to-all bandwidth. Backs up the NVLink-5 bandwidth numbers in `02_topology_mapping.md` §2 and `03_hierarchical_topologies.md` §1.1–§1.2. <https://www.nvidia.com/en-us/data-center/nvlink/>

**[UCIE]** UCIe Consortium. (2023–2024). *Universal Chiplet Interconnect Express (UCIe) Specification v1.1 / v2.0.* → Standard die-to-die interface for chiplet interconnects; per-lane signaling rates up to 32 GT/s (v1.1 and v2.0; 48 / 64 GT/s arrive with v3.0 in 2025), target latency ~2 ns per hop for advanced package PHY. Backs up the chiplet-mesh $\alpha$ and BW numbers in `02_topology_mapping.md` §1.3 and §4.1. <https://www.uciexpress.org/specifications>

**[NCCL-TESTS]** NVIDIA Corporation. (2020–2025). *NCCL Performance Tests* (`doc/PERFORMANCE.md`). → Defines the algbw / busbw metrics and the per-collective busbw factors (AllReduce $2(n-1)/n$; ReduceScatter, AllGather and AlltoAll $(n-1)/n$; Broadcast and Reduce 1). It publishes no reference measurements. Backs up the busbw/algbw vocabulary in `01_collective_algorithms.md` Appendix D. <https://github.com/NVIDIA/nccl-tests>

**[H100-SPEC]** NVIDIA Corporation. (2022). *NVIDIA H100 Tensor Core GPU Architecture.* NVIDIA Whitepaper WP-10792-001. → H100 SXM5 NVLink 4.0: 900 GB/s bidirectional per GPU (450 GB/s unidirectional). Baseline "pre-NVLink5" BW numbers behind the calibration in `05_contention_and_congestion.md` §2.1. <https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet>

**[QX800]** NVIDIA Corporation. (2024). *NVIDIA Quantum-X800 InfiniBand Switches* (Q3200-RA / Q3400-RA datasheet and product page). → Q3400-RA: 144 ports at 800 Gb/s (XDR) across 72 OSFP cages, 115.2 Tb/s throughput; a two-level fat-tree of these switches connects up to 10,368 NICs; the fourth generation of SHARP "adds support for FP8 precision and new collective operations, such as ReduceScatter and ScatterGather". The compute-fabric switch of the DGX GB300 SuperPOD [DGX-SUPERPOD-GB300]. Backs up the 144-port ToR radix in `03_hierarchical_topologies.md` §1.1 and Appendix A, and the SHARPv4 notes in `04_in_network_collectives.md` §1.3 and §2.2. <https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/>

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
| `[MAGPIE]` | `03_hierarchical_topologies.md` §2 |
| `[ZGGT09]` | `03_hierarchical_topologies.md` §2 |
| `[HAN]` | `03_hierarchical_topologies.md` §2 |
| `[TPU-V4]` | `02_topology_mapping.md` §3.4, §3.6; `05_contention_and_congestion.md` §4.1 |
| `[TRN2-ARCH]` | `02_topology_mapping.md` §3.4, §3.6 |
| `[NEURON-CC]` | `02_topology_mapping.md` §3.4, §5.1, App. A |
| `[DGX-SUPERPOD]` | `03_hierarchical_topologies.md` §1 |
| `[DGX-SUPERPOD-GB200]` | `03_hierarchical_topologies.md` §1.1, §1.2, App. A.4; `04_in_network_collectives.md` §2.2 |
| `[DGX-SUPERPOD-GB300]` | `03_hierarchical_topologies.md` §1.1, §1.2, App. A |
| `[NCCL-PXN]` | `03_hierarchical_topologies.md` §1.1, §2.2 |
| `[META-RSC]` | `03_hierarchical_topologies.md` §1 |
| `[NCCL-TESTS]` | `01_collective_algorithms.md` App. D; `05_contention_and_congestion.md` §4.1 |
| `[SHARP-IB]` | `04_in_network_collectives.md` §2.2 |
| `[NVLINK-SHARP]` | `04_in_network_collectives.md` §1.1, §2.1 |
| `[NCCL-NVLS-REPORTS]` | `03_hierarchical_topologies.md` §3; `04_in_network_collectives.md` §1.4, §4; `05_contention_and_congestion.md` §2.1, §4.1 |
| `[NCCL-NVLS-2.27]` | `04_in_network_collectives.md` §2.2 |
| `[TH-ULTRA]` | `04_in_network_collectives.md` §1.1, §1.2, §1.3, §2.1 |
| `[NVLINK5-SPEC]` | `02_topology_mapping.md` §2; `03_hierarchical_topologies.md` §1.1, §1.2 |
| `[UCIE]` | `02_topology_mapping.md` §1.3, §4.1 |
| `[H100-SPEC]` | `05_contention_and_congestion.md` §2.1 |
| `[QX800]` | `03_hierarchical_topologies.md` §1.1, App. A; `04_in_network_collectives.md` §1.3, §2.2 |

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
| NVLS busbw ≈ 470 GB/s, ≈ 1.3× over ≈ 360 GB/s without NVLS | `[NCCL-NVLS-REPORTS]` — two user reports | Reported by users, not published by NVIDIA; indicative |
| Tomahawk Ultra 250 ns switch latency | `[TH-ULTRA]` | ✓ |
| Tomahawk Ultra in-network collectives: AllReduce, Broadcast, AllGather | `[TH-ULTRA]` | ✓ (no all-to-all is listed) |
| Crossbar $\eta_\beta \approx 0.80$ (360/450) | `[NCCL-NVLS-REPORTS]` — the reporter's stated practical maximum | Reported by a user; `[NCCL-TESTS]` itself publishes no numbers |
| Torus $\eta_\beta \approx 0.60$ from twisted-torus 1.63× | `[TPU-V4]` §V | ✓ (upper-bound interpretation) |
| DGX GB300 SuperPOD: ConnectX-8 at 800 Gb/s, 144-port Q3400, two leaves per rail per SU, 32 leaf + 18 spine at 2304 GPUs, spines shared across rails | `[DGX-SUPERPOD-GB300]` text, Table 3, Figure 8 | ✓ |
| DGX GB200 SuperPOD: ConnectX-7 and QM9700 at 400 Gb/s, one leaf per rack per rail, core tier from 2 SUs | `[DGX-SUPERPOD-GB200]` | ✓ |
| SHARP on Spectrum-X Ethernet | no NVIDIA source found | ✗ removed from the series |
| Hardware A2A on a shipping switch | `[TH-ULTRA]` names AllReduce / Broadcast / AllGather; `[QX800]` lists an undefined SHARPv4 "ScatterGather" | Not documented as an all-to-all collective; modeled as a what-if |

### Known numerical caveats (documented in-line)

- Switch cut-through latencies (100–200 ns for NVSwitch, 200–400 ns or a rounded 0.5 μs for InfiniBand) are estimates: no vendor document read for this series states them, except Tomahawk Ultra's 250 ns `[TH-ULTRA]`. "NVSwitch Gen3 / Gen4" are this series' shorthand for the H100-era and NVL72-era NVLink switches; NVIDIA's NCCL guide speaks of NVLS on "NVLink Version 3 switches and later" `[NVLINK-SHARP]`.
- The $\alpha$ values used in worked examples ($\alpha = 0.5\\,\mu$s intra-NVLink; $\alpha \approx 2\\,\mu$s / $8\\,\mu$s for the same-ToR / via-spine InfiniBand paths of `03_hierarchical_topologies.md` §1.2) are order-of-magnitude estimates; 0.5 μs corresponds to NVSwitch cut-through (~100 ns) plus endpoint software (~400 ns). Exact values depend on the NCCL path chosen at runtime. The model's conclusions are robust to $\pm 2\times$ variation on $\alpha$ at bulk payloads, where the bandwidth term dominates — above roughly 1 MB for INC AR and 4 MB for DBT AR at $N = 512$ (`04_in_network_collectives.md` §3.2); ring AR at $N = 512$ stays α-dominated up to about 230 MB.
- The A2A on star entries in `02_topology_mapping.md` §2.5 and §5.1 use the per-rank pairwise direct-send formula (NVSwitch aggregate bisection saturated), appropriate for MoE workloads where all ranks participate simultaneously.
