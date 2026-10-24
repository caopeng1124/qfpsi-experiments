"""
Check stored results against the numbers the paper quotes.

A reviewer runs this instead of trusting the README: every assertion is a value
from the paper (or from the artifact summary), with a tolerance. Accuracy numbers
are deterministic given the seeds, so their tolerance is tight; wall-clock
numbers depend on the machine and are only checked for order of magnitude.

Usage:
  python scripts/check_results.py                          # published results
  python scripts/check_results.py --results-dir experiments/results_quick --loose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = []      # (name, file, path, expected, rel)
CUSTOM = []      # (name, file, fn(data) -> (ok, description))


def check(name, fname, path, expected, rel=0.10):
    CHECKS.append((name, fname, path, expected, rel))


def custom(name, fname, fn):
    CUSTOM.append((name, fname, fn))


def get(d, path):
    """Look up "a/b/0.01/c": "/" separates, dots are literal key characters."""
    cur = d
    for part in path.split("/"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return float(cur)


# --- E1: the analytic overlap law -------------------------------------------
check("E1 overlap rho=.85 w=1 (B=1)", "predicate_multiseed.json", "aggregate/B1w1/mean/9", 105.5, 0.02)
check("E1 overlap rho=.85 w=2 (B=1)", "predicate_multiseed.json", "aggregate/B1w2/mean/9", 86.8, 0.02)
check("E1 overlap rho=.85 w=8 (B=1)", "predicate_multiseed.json", "aggregate/B1w8/mean/9", 26.9, 0.05)
check("E1 d' far (B,w)=(2,2)", "predicate_multiseed.json", "aggregate/B2w2/dprime", 4.76, 0.06)
check("E1 old default (4,8) FNMR", "predicate_multiseed.json",
      "aggregate/B4w8/ops/0.01/fnmr", 0.938, 0.05)

# --- E2: ablations (rows are ordered config-major, eps-minor) ---------------
check("E2 default FNMR@1%FMR", "ablations.json", "rows/0/fnmr", 0.0067, 0.30)
check("E2 expansion FNMR@1%FMR", "ablations.json", "rows/2/fnmr", 0.0004, 0.60)
check("E2 expansion T_q/T", "ablations.json", "rows/2/tq_ratio", 1.61, 0.05)

# --- E3: near-miss resolution ------------------------------------------------
check("E3 near-miss release rate", "boundary.json", "rows/0/fmr_0.8_test", 0.9496, 0.05)
check("E3 near-miss d' (2,2,128)", "boundary.json", "rows/0/dprime_0.8", 0.74, 0.15)

# --- E4: transfer curve ------------------------------------------------------
check("E4 release prob at cos=0.60", "transfer.json", "configs/B2_w2_T128/mean/2", 0.30, 0.20)
check("E4 release prob at cos=0.70", "transfer.json", "configs/B2_w2_T128/mean/4", 0.66, 0.10)

# --- E5: selectivity (row 4 is (2,8,512)) -----------------------------------
check("E5 bucket at 1e6 (2,8,512)", "selectivity.json", "rows/4/bucket_at_1e6", 15.3, 0.05)
check("E5 scanned/query at 1e6", "selectivity.json", "rows/4/scanned_at_1e6", 7812, 0.05)
check("E5 near-miss d' (2,8,512)", "selectivity.json", "rows/4/dprime_near", 0.93, 0.12)

# --- E7: real data -----------------------------------------------------------
check("E7 SIFT overlap floor", "real_sift.json", "rows/0/overlap_far", 55.1, 0.10)
check("E7 SIFT FNMR@1%FMR", "real_sift.json", "rows/0/fnmr_close", 0.41, 0.25)

# --- E8/E9: systems (order of magnitude) ------------------------------------
check("E8 scalar mult median (ms)", "oprf_bench.json", "scalarmult/median_ms", 0.0325, 0.60)
check("E9 latency median (ms)", "proto_10k.json", "latency_ms/median", 18.2, 0.60)
check("E9 bytes per query", "proto_10k.json", "bytes/total_median", 13986, 0.05)
check("E9 hits per query", "proto_10k.json", "hits_median", 80, 0.05)

# --- E10: executed two-stage -------------------------------------------------
check("E10 recovered (i.i.d.)", "twostage_proto.json", "results/iid/planted_recovered", 0.97, 0.10)
check("E10 false positives (clustered)", "twostage_proto.json",
      "results/clustered/false_pos_mean", 25.07, 0.40)

# --- E11: adaptive collapses to one width -----------------------------------
def adaptive_is_constant(d, loose):
    hist = d["results"]["adaptive"]["B_hist"]
    n = sum(hist.values())
    share = max(hist.values()) / n
    need = 0.80 if loose else 0.99
    return share >= need, f"dominant width {max(hist, key=hist.get)} over {share:.0%} of {n} queries"


custom("E11 adaptive collapses to one width", "adaptive_proto.json", adaptive_is_constant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--loose", action="store_true",
                    help="triple every tolerance (quick runs use smaller samples)")
    args = ap.parse_args()
    res = Path(args.results_dir)
    slack = 3.0 if args.loose else 1.0

    failures = skipped = 0
    width = max(len(c[0]) for c in CHECKS + [(n, None, None, None, None) for n, _, _ in CUSTOM])

    for name, fname, path, expected, rel in CHECKS:
        f = res / fname
        if not f.exists():
            print(f"   SKIP {name:<{width}}  ({fname} missing)")
            skipped += 1
            continue
        try:
            got = get(json.loads(f.read_text()), path)
        except (KeyError, IndexError, ValueError) as e:
            print(f"   FAIL {name:<{width}}  ({fname}:{path}: {e})")
            failures += 1
            continue
        tol = abs(expected) * rel * slack
        ok = abs(got - expected) <= tol
        print(f"   {'ok  ' if ok else 'FAIL'} {name:<{width}}  got {got:>11.5g}  "
              f"expected {expected:>10.4g} ±{tol:.3g}")
        failures += 0 if ok else 1

    for name, fname, fn in CUSTOM:
        f = res / fname
        if not f.exists():
            print(f"   SKIP {name:<{width}}  ({fname} missing)")
            skipped += 1
            continue
        ok, desc = fn(json.loads(f.read_text()), args.loose)
        print(f"   {'ok  ' if ok else 'FAIL'} {name:<{width}}  {desc}")
        failures += 0 if ok else 1

    print()
    if failures:
        print(f"   {failures} check(s) failed, {skipped} skipped")
        return 1
    print(f"   all {len(CHECKS) + len(CUSTOM) - skipped} checks passed "
          f"({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
