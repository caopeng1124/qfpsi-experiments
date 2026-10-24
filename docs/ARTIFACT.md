# Artifact guide

This document is written for a reviewer who wants to see the measurements rather
than read about them. It states what to run, how long it takes, what output to
expect, and which paper claim each experiment supports.

---

## 1. What this artifact is

`experiments/` contains the code that produced every number in the evaluation
section of the paper. There are three kinds of numbers in the paper, and the
artifact makes the difference explicit:

| Class | Produced by | Checked how |
|---|---|---|
| Analytical | the protocol format (e.g. `53NT` bytes per entry) | by reading the code that computes it |
| Measured | the scripts here | `scripts/check_results.py` re-derives them from the stored JSON |
| Reported | quoted from prior papers | not reproduced; cited in the paper |

## 2. Environment

```
macOS 27 / Apple M3 (8 cores, 16 GB), Python 3.14.7
numpy 2.5.4, matplotlib 3.11.2, h5py 3.16.0, libsodium 1.0.22
```

Linux is fine too; the only platform-specific part is `experiments/proto/netem_dnctl.sh`
(kernel traffic shaping, optional). Install:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
brew install libsodium            # Debian/Ubuntu: sudo apt install libsodium23
```

## 3. Run everything

```bash
./run_all.sh            # quick mode, ~4-5 minutes, reduced sample sizes
./run_all.sh --full     # full mode, ~20 minutes, reproduces the published numbers
```

Each step logs to `logs/`, writes JSON to `experiments/results/`, and the script
finishes with `scripts/check_results.py`, which asserts the stored results
against the values quoted in the paper. A correct run ends with:

```
   all 22 checks passed
```

Real data (experiment E7) is optional and skipped unless the HDF5 file is
present:

```bash
./scripts/get_data.sh   # SIFT1M, 525 MB, resumable
./run_all.sh
```

## 4. What each experiment shows

| ID | Question the reviewer should ask | Script | Paper claim it backs |
|---|---|---|---|
| E1 | Does token overlap track similarity as the model predicts? | `sim/run_multiseed.py` | §6.2, Table 2: `E[ov] ≈ T·p_c^w`, and the previous default `(B,w)=(4,8)` is inoperable |
| E2 | How do expansion, token count and dimension move the operating point? | `sim/run_ablations.py` | §6.2, Table 3 |
| E3 | Can the predicate tell a genuine match from a near miss? | `sim/run_boundary.py` | §6.3, Table 4: `d' < 1` everywhere |
| E4 | What decision rule does the calibrated predicate actually implement? | `sim/run_transfer.py` | §6.3, Fig. 6: the soft threshold |
| E5 | What does selectivity cost, and does it fix the boundary? | `sim/run_selectivity.py` | §6.3: bucket size `N/2^{Bw}`, `w=8` recommended at `N=10^6` |
| E6 | How large is the candidate set? | `sim/run_twostage.py` | §6.3 (count-only cross-check of E10) |
| E7 | Do the conclusions survive real descriptors? | `sim/run_realdata.py` | §6.2 "Real descriptors": overlap floor 55/128, FNMR 40% |
| E8 | How fast is the OPRF? | `proto/bench_oprf.py` | §6.5, Table 7 |
| E9 | What is the end-to-end latency and traffic? | `proto/qfpsi_proto.py` | §6.5: 18.2 ms, 13,986 B; `--shaping fast/slow` for LAN/WAN |
| E10 | Does the two-stage protocol work end to end? | `proto/twostage_proto.py` | §6.3, Table 5: 97% recovery, 25 false positives/query on clustered data |
| E11 | Does Adaptive-`B` help? | `proto/adaptive_proto.py` | §6.3 "Adaptive-B, executed": it degenerates to a constant |

## 5. Expected results (so you can compare)

Accuracy numbers are deterministic given the seeds, so they should match closely;
wall-clock numbers depend on the machine and are only checked for order of
magnitude.

| Quantity | Expected | Source |
|---|---|---|
| Overlap at `ρ=0.85`, `B=1`, `w=1/2/4/8` | 105.5 / 86.8 / 58.9 / 26.9 | E1 |
| `(B,w)=(4,8)`: FNMR at 1% FMR | 93.8% | E1 |
| Expansion `r=1`: FNMR 0.04%, `T_q/T` 1.61 | E2 |
| Near-miss release rate at `ρ=0.80` | 94.96% | E3 |
| Release probability at `cos=0.60 / 0.70 / 0.80` | 30% / 66% / 95% | E4 |
| Bucket size at `N=10^6`, `(2,8,512)` | 15 entries | E5 |
| SIFT overlap floor (unrelated records) | 55/128 | E7 |
| SIFT FNMR at 1% FMR | ~40% (`(2,2,128)`), ~26% (`(2,8,512)`) | E7 |
| Scalar multiplication | ~32 µs | E8 |
| Query latency, `N=10^4`, one core | ~18 ms | E9 |
| Bytes per query, 80 hits | 13,986 | E9 |
| Planted label recovered (two-stage) | 97% | E10 |
| Adaptive-`B` width histogram | `{2: 32}` (constant) | E11 |

## 6. Known limitations of the artifact

- The prototypes are Python; the paper says so and does not compare them with
  compiled implementations.
- `proto/qfpsi_proto.py --shaping` emulates the network in user space (sleeps
  proportional to bytes and RTT). Kernel-level shaping is provided in
  `proto/netem_dnctl.sh` but needs root and was not part of the reported runs.
- Only the fixed-`B`, two-stage and Adaptive-`B` paths are executed. Malicious
  security, PIR retrieval, and the GIST/Deep/face workloads are not implemented.
- E7 needs a 525 MB download; the rest of the artifact runs offline.

## 7. Re-deriving the data package

`experiments/make_dataset.py` regenerates `results/csv/*.csv`,
`results/MANIFEST.json` (script, row count, size, sha256 prefix per file) and
`results/SUMMARY.md` (the map from paper tables to data files).
