# Handover

Working notes for picking this project up on another machine or in another session. Not part of the published book.

**Norm:** this file is rewritten at the end of every work round and pushed with it, so `main` always carries the
current state. Read it first; update it last.

Last updated: 2026-11-15, at commit `57c261e` plus the commit that adds this file.

## State

- `main` is the only branch, clean and pushed. A push to `main` rebuilds and deploys the mdBook site
  (`.github/workflows/mdbook.yml`), so confirm before pushing.
- The last round ("spec check") is complete: three commits, `34c4472`, `452bf85`, `57c261e`. It went through three
  independent reviews; all findings are applied. Tests (21), the anchor self-test and the book build pass.
- Nothing is in progress. What remains is under "Open decisions" and "Not started".

## What the last round changed

1. **Exact anchors.** The torus anchor is printed as 56.5 μs (exact 56.49). `canonical_anchors()` pins all seven
   anchors to two decimals (546.49, 44.56, 18.78, 56.49, 35.19, 53.44, 84.34 μs; `ANCHOR_TOLERANCE_US = 0.005`); the
   docs print them rounded.
2. **Worked SuperPOD is NVIDIA's DGX GB300 reference architecture** (Quantum-X800 Q3400, ConnectX-8, 800 Gb/s), so
   `BW_outer` is 100 GB/s and every case-study number in `03_hierarchical_topologies.md` was recomputed (2-pod AR
   112.3 μs; A2A 268 μs; L = 32 ladder 126.5 / 46.3 / 40 μs). Appendix A follows NVIDIA's counts: 32 leaves + 18
   spines, spine layer shared by all four rails, 4 cables per leaf–spine pair (inferred from the counts and labelled so).
   The GB200 design (QM9700, NDR, 50 GB/s) appears only as a comparison.
3. **Hardware claims re-sourced.** "Spectrum-X SHARP" removed. Hardware all-to-all is a what-if primitive, worded as
   "not documented": Tomahawk Ultra names AllReduce / Broadcast / AllGather, and the Quantum-X800 datasheet lists an
   undefined SHARPv4 "ScatterGather". The η_β values 0.80 / 0.52 are attributed to two community GitHub reports
   (`[NCCL-NVLS-REPORTS]`) and described as indicative.
4. **Defaults.** `util.Anchors` outer tier is 2 μs / 100 GB/s. Notebook 04's optimum is L = 8 (exact tie with L = 16
   at equal tier bandwidths). README example 3 uses 4 pods (121.4 μs).
5. **Title.** The H1 of `03_hierarchical_topologies.md` is now "Hierarchical Topologies: Clos Fabrics, Composition, and
   In-Network Collectives" (file name unchanged).

## Open decisions (owner's call; do not change without asking)

- **η_β calibration pair.** (0.80, 0.52) reads the reports with a busbw factor of 2. With the exact 8-GPU factor 1.75
  the equally consistent pair is (0.91, 0.60). Switching would move the realistic anchors (INC 35.19 → about 30.8 μs,
  DBT 53.44 → about 47.9 μs). The docs keep (0.80, 0.52) and state the caveat in `05_contention_and_congestion.md` §4.1.
- **Title rename** of `03_hierarchical_topologies.md` (item 5) has not been confirmed.
- **Old git history** of notebook 00 contains a printed home-directory path. Removing it needs a history rewrite and a
  force push.

## Not started

- **Downstream mirror.** `spiceMonkey/flare-llm` vendors `code/core/collective_cost.py` and `documentation/modeling/`
  (its `.github/workflows/sync-collectives.yml`). Its copy matches this repo at `afca6db`, so it needs a re-sync. No
  cost function changed numerically in the last round. Its own text also repeats two claims this repo dropped
  (Spectrum-X SHARP; hardware A2A on Tomahawk Ultra / Rubin) in `llm_perf/specs/system_spec.py`,
  `llm_perf/io/system_loaders.py`, `llm_perf/core/primitives/dispatch.py`, `documentation/api/specs.md` and
  `documentation/modeling/references.md`. Never edit that project from a session working on this one.
- **Paper drafts** (local only, see below) still carry the claims corrected in the last round.
- `[H100-SPEC]` in `references.md` links to a URL that now redirects to NVIDIA's homepage; no verified replacement.

## What is NOT in the repo (missing on a fresh clone)

`.gitignore` excludes these, so they exist only on the original machine:

- `CLAUDE.md` — project instructions for Claude Code. The conventions section below repeats what matters.
- `scratch/build_notebook_0X.py` and `scratch/build_topologies_figure.py` — the generators for the five notebooks and
  for `assets/topologies_modeled.svg`. **Without them the notebooks can only be edited by hand.** Copy `scratch/` across
  before changing a notebook or that figure.
- `scratch/paper/` — the tutorial-paper drafts; `scratch/round6_scripts/` — the last round's edit scripts.
- `.venv/`.

## Setup on a new machine

```bash
git clone https://github.com/spiceMonkey/collective-comm.git && cd collective-comm
python3 -m venv .venv                                  # Python 3.14 on the original machine
.venv/bin/pip install jupyterlab matplotlib pandas     # only for notebooks and charts; core code is stdlib-only
# mdbook 0.5.2 for the book (same version as the deploy workflow)
```

## Checks to run after any change

```bash
.venv/bin/python -m unittest discover -s code/tests
cd code && ../.venv/bin/python core/collective_cost.py && cd ..
mdbook build -d /tmp/collective-book
```

After changing a cost function or a notebook (needs the `scratch/` builders): regenerate all five notebooks, execute
all five with `.venv/bin/jupyter execute --inplace code/notebooks/0*.ipynb`, run
`.venv/bin/python code/notebooks/export_figures.py`, then re-read the notebooks' "Key takeaways" and the chart captions
in the docs against the new output.

## Conventions (decided; keep docs, code, notebooks and README consistent)

- `documentation/modeling/` is the specification. Code mirrors its equations.
- **BW is the per-rank port bandwidth**, one direction.
- **DBT all-reduce on a single port has n_β = 2**: pipelined `2⌈log₂N⌉·α + 2·M/BW`. `multi_port=True` halves the BW terms.
- **Every in-network primitive costs `2·α_switch`** on one switch, `2k·α_switch` across `k` tiers. Only all-reduce gets
  a bandwidth lift (`2M/BW → M/BW`).
- **Ring-relay A2A** is `⌊N/2⌋·α + (N/8)·M/BW`. Torus A2A coefficient is `d_max/8` (a size-2 dimension counts 1/2);
  mesh is `d_max/4`.
- **Hierarchical composition:** AR ≡ inner RS → outer AR on `M·L/N` → inner AG; AG ≡ outer AG → inner AG;
  RS ≡ inner RS → outer RS. AG and RS are exactly half of AR.
- **Markdown math:** write `^{\ast}` / `_{\ast}`, `\_` after `}` or `)`, and `\\,` (two backslashes) for a thin space,
  or mdBook mangles it.
- **Sourcing:** say "not documented" rather than "does not exist"; label anything inferred from a vendor figure as
  inferred; community-reported numbers are called indicative.
- **Process:** after a substantial fix pass, run one or two fresh independent reviewers before calling it done. Work on
  a short branch, fast-forward `main`, delete the branch.
- **Interface:** the downstream mirror calls the cost functions positionally as `(M, G, alpha_s, bw_Bps)`; treat a
  signature or numeric change here as an interface change and say so.

## Sources behind the last round

- GB300 reference architecture: <https://docs.nvidia.com/pdf/dgx-spod-gb300-ra.pdf> (compute fabric: Figure 8, Table 3)
- GB200 reference architecture: <https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html>
- Quantum-X800: <https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/> (datasheet: 10,368 NICs, SHARPv4)
- Tomahawk Ultra release: <https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-tomahawk-ultra-reimagining-ethernet-switch-hpc>
- NCCL PXN: <https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/>
- Community NVLS numbers: <https://github.com/NVIDIA/nccl-tests/issues/312>, <https://github.com/NVIDIA/nccl/issues/1221>
