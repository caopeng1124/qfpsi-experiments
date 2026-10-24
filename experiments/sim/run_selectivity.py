"""
M10: accuracy under the selectivity constraint.

The executed two-stage run showed that the index bucket size is N/2^{Bw}, so the
token width B*w must satisfy 2^{Bw} >> N for the lookup to stay selective.
Accuracy, however, prefers small w. This script measures both sides of that
trade-off at the same parameter points:

  accuracy : overlap distribution, d' against unrelated pairs and against
             near misses (cos 0.80/0.82), FMR at 99% genuine recall, FNMR at a
             1% false-match budget
  selector : token space 2^{Bw}, expected bucket size N/2^{Bw} at N=10^6, and
             the expected number of index entries scanned per query, T*N/2^{Bw}

Five rotation seeds, 2000 pairs per similarity level, calibration and test sets
drawn independently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402

RHO = [0.0, 0.5, 0.80, 0.82, 0.85]
N_TARGET = 10 ** 6


def run(seed, d, B, w, T, n):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    sg = 1.0 / np.sqrt(d)
    zmin, zmax = -3 * sg, 3 * sg
    W = sim.windows(rng, d, T, w)

    def ov(rho, count):
        q, x = sim.make_pairs(rng, rho, count, d)
        U = sim.quantize(sim.encode(np.vstack([q, x]), P, d, zmin, zmax), B, zmin, zmax)
        tok = sim.tokenize(U, W, B)
        return sim.overlap(tok[:count], tok[count:]).astype(float)

    cal = {r: ov(r, n) for r in RHO}
    test = {r: ov(r, n) for r in RHO}

    t99 = int(np.quantile(cal[0.85], 0.01))           # 99% genuine recall
    op = sim.operating_point(cal[0.85][:0], cal[0.5], T, 1e-2)  # placeholder
    # threshold meeting FMR<=1% against unrelated pairs (calibration set)
    t_far = next(t for t in range(T + 1)
                 if float((cal[0.5] >= t).mean()) <= 1e-2)

    def dprime(a, b):
        den = np.sqrt((test[a].var() + test[b].var()) / 2)
        return float((test[a].mean() - test[b].mean()) / den) if den > 0 else 0.0

    return {
        "overlap_085": float(test[0.85].mean()),
        "overlap_080": float(test[0.80].mean()),
        "overlap_050": float(test[0.50].mean()),
        "t99": t99, "t_far1": t_far,
        "fnmr_at_t99": float((test[0.85] < t99).mean()),
        "fmr080_at_t99": float((test[0.80] >= t99).mean()),
        "fmr082_at_t99": float((test[0.82] >= t99).mean()),
        "fmr050_at_t99": float((test[0.50] >= t99).mean()),
        "fnmr_at_tfar": float((test[0.85] < t_far).mean()),
        "fmr050_at_tfar": float((test[0.50] >= t_far).mean()),
        "dprime_far": dprime(0.85, 0.5),
        "dprime_near": dprime(0.85, 0.80),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/results/selectivity.json")
    args = ap.parse_args()

    configs = [(2, 2, 128), (2, 4, 128), (2, 6, 256), (2, 8, 256),
               (2, 8, 512), (2, 10, 512), (2, 12, 512), (1, 12, 512)]
    rows = []
    for B, w, T in configs:
        res = [run(s, 128, B, w, T, args.n) for s in args.seeds]
        row = {"B": B, "w": w, "T": T, "token_space": 2 ** (B * w),
               "bucket_at_1e6": N_TARGET / 2 ** (B * w),
               "scanned_at_1e6": T * N_TARGET / 2 ** (B * w)}
        for k in res[0]:
            row[k] = float(np.mean([r[k] for r in res]))
            row[k + "_sd"] = float(np.std([r[k] for r in res]))
        rows.append(row)
        print(f"done B={B} w={w} T={T}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": args.n, "seeds": args.seeds,
                                          "rho": RHO, "N_target": N_TARGET,
                                          "rows": rows}, indent=1))

    print("\n| B | w | T | 2^Bw | bucket@1e6 | scanned/query@1e6 | ov@.85 | d'(far) | d'(near) | FMR@0.80 (99% recall) | FNMR@1%FMR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['B']} | {r['w']} | {r['T']} | {r['token_space']:.0f} | "
              f"{r['bucket_at_1e6']:.1f} | {r['scanned_at_1e6']:.0f} | "
              f"{r['overlap_085']:.1f} | {r['dprime_far']:.2f} | {r['dprime_near']:.2f} | "
              f"{100*r['fmr080_at_t99']:.1f}% | {100*r['fnmr_at_tfar']:.1f}% |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
