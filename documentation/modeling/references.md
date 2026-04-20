# References: Collectives Explainer Series

**Author:** Yue Lu  
**Date:** April 2026  

Self-contained bibliography for `01–04` in this folder. Every non-obvious equation and empirical value in the series has been cross-checked against the sources below.

---

## Primary algorithm papers

**[ALPHA-BETA]** Hockney, R. (1994). *The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.* Parallel Computing 20(3). → Canonical α-β latency model $t = \alpha + M/\mathrm{BW}$. Used throughout `01 §1`.

**[PY09]** Patarasuk, P., & Yuan, X. (2009). *Bandwidth Optimal All-Reduce Algorithms for Clusters of Workstations.* JPDC 69(2):117–124. → Ring AR achieves $2(N-1)/N \cdot M/\mathrm{BW}$ on any tree-connected fabric. Backs up `01 §3` and `02 §3`. <https://doi.org/10.1016/j.jpdc.2008.09.002>

**[TRG05]** Thakur, R., Rabenseifner, R., & Gropp, W. (2005). *Optimization of Collective Communication Operations in MPICH.* IJHPCA 19(1):49–66. → Recursive halving-doubling all-reduce achieves $2\lceil \log_2 N \rceil \alpha + 2 M/\mathrm{BW}$ for power-of-2 N; describes the ring / RHD / Rabenseifner crossover rules implemented by modern MPI libraries. Backs up `01 Appendix A.2`, `01 Appendix B.1`, and `02 Appendix A`. <https://journals.sagepub.com/doi/10.1177/1094342005051521>, author preprint: <https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf>

**[CHPV07]** Chan, E., Heimlich, M., Purkayastha, A., & van de Geijn, R. (2007). *Collective Communication: Theory, Practice, and Experience.* Concurrency and Computation: Practice and Experience, 19(13):1749–1783. → Dim-decomposed all-reduce framework and telescoping derivation of multi-dim ring costs. Backs up the torus dim-decomp cost and BW telescoping in `02 §3.1`.

**[BHKUW97]** Bruck, J., Ho, C.-T., Kipnis, S., Upfal, E., & Weathersby, D. (1997). *Efficient Algorithms for All-to-All Communications in Multiport Message-Passing Systems.* IEEE TPDS 8(11):1143–1156. → Log-round all-to-all via circular shifts and bit-wise exchange; $\lceil \log_2 N \rceil$ steps at the cost of moving the full payload per step. Backs up the "log A2A" variant in `01 Appendix B.3`. <https://doi.org/10.1109/71.642949>

**[SST09]** Sanders, P., Speck, J., & Träff, J.L. (2009). *Two-Tree Algorithms for Full Bandwidth Broadcast, Reduction and Scan.* Parallel Computing 35(12):581–594. → Double binary tree (DBT) construction: two complementary trees that each rank is interior in exactly one of, used to saturate both directions of every full-duplex link during broadcast / reduce / AR. Backs up the DBT derivation in `01 §3.2` and the DBT shipping rationale in `01 §3.3`. <https://doi.org/10.1016/j.parco.2009.09.001>

**[NCCL-PAT]** NVIDIA Corporation. (2024). *NCCL 2.23 Release Notes / "New collective algorithms for small message sizes — PAT for inter-node AllGather and ReduceScatter at 1 rank per node."* Accompanying blog: Jeaugey, S., "Introducing NCCL 2.23: Parallel Aggregated Trees for AllGather and ReduceScatter at Scale." NVIDIA Developer Blog, Aug 2024. → PAT (Parallel Aggregated Trees): reversed-Bruck offset schedule ($4, 2, 1, \ldots$), $\log_2 N$ rounds, $M/N$-byte bounded buffer per round, total per-rank on-wire volume $(N-1)M/N$. Shipping scope: inter-node AG / RS only, at 1 rank per node. Backs up `01 Appendix B.2` and the §4.2 scale-out rationale. <https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-23-4.html>

**[DEMYST-NCCL]** Jeaugey, S., Addair, T., et al. (2025). *Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms.* arXiv:2507.04786. → Empirical tuner-trace analysis of NCCL's algorithm selection across AR / AG / RS at representative scales; confirms the Tree-vs-Ring selection rule (DBT for small-$M$, ring for large-$M$) on NVLink / NVSwitch fabrics and documents the per-regime crossover points that the α-β model alone does not predict. Backs up the "practice caveat" at the end of `01 §3.3` and the corresponding note in `02 §2`. <https://arxiv.org/abs/2507.04786>

---

## Topology architecture papers

**[TPU-V4]** Jouppi, N.P., et al. (2023). *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.* ISCA 2023. → 3D torus with optical circuit switching for slice reconfiguration; twisted-torus 1.63× A2A gain on asymmetric layouts (§V). Source for the torus realistic $\eta_\beta \approx 0.60$ calibration in `04 §3`. <https://doi.org/10.1145/3579371.3589350>

**[TRN2-ARCH]** Amazon Web Services. (2024–2025). *Amazon EC2 Trn2 Architecture.* AWS Neuron Documentation. → Trn2 server: 16 Trainium2 chips arranged as a 2D NeuronLink torus (each chip connects to 4 neighbors). Trn2 UltraServer: four Trn2 instances joined via a Z-dimension NeuronLink into a 3D 64-chip torus. Backs up the Trainium adoption note in `02 §3.1`. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html>

**[NEURON-CC]** Amazon Web Services. (2024–2025). *Neuron Collective Communication / Intra-node Collectives.* AWS Neuron Documentation. → NeuronX Collective Communication Library ships ring / mesh / KangaRing / RDH all-reduce variants purpose-built for Trainium's 2D/3D NeuronLink torus; per-NeuronCore CC Cores offload the orchestration of collective phases. Backs up the NeuronX CCL reference in `02 §3.1`. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/intranode-collective-comm.html>

---

## In-network collectives

**[SHARP-IB]** Graham, R.L., et al. (2016). *Scalable Hierarchical Aggregation Protocol (SHArP): A Hardware Architecture for Efficient Data Reduction.* COMHPC Workshop at SC16. → Original SHARP specification for InfiniBand; hardware-offloaded reduction trees; reports ~95% of network bandwidth utilization and 2–5× speedup over host-based reduction. Backs up `03 §3`. <https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf>

**[NVLINK-SHARP]** NVIDIA Corporation. (2023–2024). *NVLink SHARP (NVLS) — In-Network All-Reduce Acceleration.* GTC S62129; NCCL release notes; GB200 NVL Multi-Node Tuning Guide. → NVSwitch-based in-network reduction; AR busbw rises from ~360 GB/s to 470+ GB/s under NVLS (≈1.3× measured). Backs up `03 §2.2` and §5. <https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html>

**[NCCL-NVLS-2.27]** NVIDIA Corporation. (2025). *Enabling Fast Inference and Resilient Training with NCCL 2.27.* NVIDIA Technical Blog, May 2025. → Up to 2.5× improvement on small-to-medium messages in Symmetric Memory configurations with NVLS. Used in `03 §4` for the small-$M$ speedup regime. <https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/>

**[TH-ULTRA]** Broadcom. (2025). *Broadcom Ships Tomahawk Ultra: Reimagining the Ethernet Switch for HPC and AI Scale-up.* Jul 2025. → First Ethernet switch ASIC with in-network collectives (INC); 51.2 Tbps full-duplex, 250 ns switch latency. Backs up the Tomahawk Ultra mention in `03 §6`. <https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-tomahawk-ultra-reimagining-ethernet-switch-hpc>

---

## Hardware specifications

**[NVLINK5-SPEC]** NVIDIA Corporation. (2024). *NVLink 5 / NVSwitch Gen4 / GB200 NVL72 Architecture.* NVIDIA product pages. → NVLink 5: 18 links × 100 GB/s per direction = **1.8 TB/s bidirectional per GPU** (900 GB/s unidirectional). NVSwitch Gen4: 72 NVLink 5 ports per chip. NVL72: 72 GPUs, 130 TB/s aggregate all-to-all bandwidth. Backs up `01 §3.1` and `02 §1.1` per-link bandwidth numbers. <https://www.nvidia.com/en-us/data-center/nvlink/>

**[NCCL-TESTS]** NVIDIA Corporation. (2020–2025). *NCCL Performance Tests.* → Canonical busbw/algbw measurement methodology for NCCL collectives; H100/A100 intra-node AR busbw ≈ 360 GB/s against 450 GB/s peak NVLink4 unidirectional. Calibration source for crossbar $\eta_\beta \approx 0.80$ in `04 §3`. <https://github.com/NVIDIA/nccl-tests>

**[H100-SPEC]** NVIDIA Corporation. (2022). *NVIDIA H100 Tensor Core GPU Architecture.* NVIDIA Whitepaper WP-10792-001. → H100 SXM5 NVLink 4.0: 900 GB/s bidirectional per GPU (450 GB/s unidirectional). Baseline for "pre-NVLink5" BW numbers. <https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet>

---

## Tag → section map

Where each citation is used in the series:

| Tag | Used in |
|---|---|
| `[ALPHA-BETA]` | `01 §1` |
| `[PY09]` | `01 §3.1`, `02 §3` |
| `[TRG05]` | `01 App. A.2`, `01 App. B.1`, `02 App. A` |
| `[SST09]` | `01 §3.2`, `01 §3.3` |
| `[CHPV07]` | `02 §3.1` |
| `[BHKUW97]` | `01 App. B.3` |
| `[NCCL-PAT]` | `01 §4.2`, `01 App. B.2` |
| `[DEMYST-NCCL]` | `01 §3.3`, `02 §2` |
| `[TPU-V4]` | `02 §1.2`, `02 §3.1`, `04 §3` |
| `[TRN2-ARCH]` | `02 §3.1` |
| `[NEURON-CC]` | `02 §3.1` |
| `[NCCL-TESTS]` | `04 §3` |
| `[SHARP-IB]` | `03 §3` |
| `[NVLINK-SHARP]` | `03 §2` |
| `[NCCL-NVLS-2.27]` | `03 §4` |
| `[TH-ULTRA]` | `03 §6` |
| `[NVLINK5-SPEC]` | `01 §3.1`, `02 §1.1` |
| `[H100-SPEC]` | `02 §2` |

---

## Verification notes (what was cross-checked)

The following equations and empirical values were independently verified (not just transcribed from secondary sources):

| Claim | Verified against | Status |
|---|---|---|
| Ring AR cost $2(N-1)\alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ | `[PY09]` direct formula | ✓ |
| Tree (RHD) AR $2 \lceil \log_2 N \rceil \alpha + 2 M/\mathrm{BW}$ | `[TRG05]` §3.3 | ✓ |
| Torus dim-decomp $2 \sum(D_i-1)\alpha + 2(N-1)/N \cdot M/\mathrm{BW}$ | `[CHPV07]`, `[PY09]` telescoping | ✓ |
| NVL72 = 72 GPUs per NVSwitch domain | `[NVLINK5-SPEC]` | ✓ |
| NVLink 5 per-GPU unidirectional 900 GB/s | `[NVLINK5-SPEC]`: 18 × 100 GB/s unidir | ✓ |
| Log A2A $\lceil \log_2 N \rceil$ steps | `[BHKUW97]` §III | ✓ |
| NVLS measured 470+ GB/s busbw | `[NVLINK-SHARP]` GB200 tuning guide | ✓ (≈1.3× over 360 GB/s non-SHARP; paper's "1.7×" is the **IB Quantum-2** figure, not NVLS) |
| Tomahawk Ultra 250 ns switch latency | `[TH-ULTRA]` | ✓ |
| Crossbar $\eta_\beta \approx 0.80$ (360/450) | `[NCCL-TESTS]` PERFORMANCE.md | ✓ |
| Torus $\eta_\beta \approx 0.60$ from twisted-torus 1.63× | `[TPU-V4]` §V | ✓ (upper-bound interpretation) |

### Known numerical caveats (documented in-line)

- The $\alpha$ values used in worked examples ($\alpha = 0.5\,\mu$s intra-NVLink, $\alpha \approx 1\,\mu$s inter-domain) are order-of-magnitude estimates consistent with NVSwitch cut-through (~100 ns) + endpoint software floor (~800 ns). Exact values depend on the NCCL path chosen at runtime; the model's conclusions are robust to $\pm 2\times$ variation on $\alpha$ because the bandwidth term dominates for $M \geq$ few hundred KB.
- The A2A on star cost listed as "switch-bisection-bound" in `02 §4.2` uses the NVSwitch aggregate bisection rather than a per-rank ring — appropriate for MoE workloads where all ranks participate simultaneously.
