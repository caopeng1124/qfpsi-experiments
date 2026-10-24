"""
Boundary sharpness: can the predicate separate a genuine match (cos = 0.85)
from an impostor that is *almost* close (cos = 0.80 / 0.82)?

This is the accuracy question that the per-pair FMR on unrelated pairs hides.
For every configuration we report, on held-out pairs with a calibration set:
  d'                  separation between the genuine and impostor overlaps
  impostor FMR        fraction of impostors above the threshold that keeps
                      1% of genuine matches (i.e. the cost of 99% recall)
  t, genuine FNMR

Configurations vary T and w; nothing is projected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402

GENUINE = 0.85
IMPOSTORS = (0.80, 0.82)


def draw(rng, rho, n, d):
    return sim.make_pairs(rng, rho, n, d)


def run(seed, d, B, w, T, n):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    sigma = 1.0 / np.sqrt(d)
    zmin, zmax = -3 * sigma, 3 * sigma
    W = sim.windows(rng, d, T, w)

    def overlaps(rho, count):
        q, x = draw(rng, rho, count, d)
        U = sim.quantize(sim.encode(np.vstack([q, x]), P, d, zmin, zmax), B, zmin, zmax)
        tok = sim.tokenize(U, W, B)
        return sim.overlap(tok[:count], tok[count:]).astype(float)

    g_cal = overlaps(GENUINE, n)
    i_cal = {r: overlaps(r, n) for r in IMPOSTORS}
    g_test = overlaps(GENUINE, n)
    i_test = {r: overlaps(r, n) for r in IMPOSTORS}

    # threshold calibrated to keep 99% of genuine matches on the calibration set
    t = int(np.quantile(g_cal, 0.01))
    out = {"t": t, "fnmr_test": float((g_test < t).mean()),
           "genuine_mean": float(g_test.mean()), "genuine_sd": float(g_test.std())}
    for r in IMPOSTORS:
        out[f"impostor_{r}_mean"] = float(i_test[r].mean())
        out[f"impostor_{r}_sd"] = float(i_test[r].std())
        out[f"fmr_{r}_test"] = float((i_test[r] >= t).mean())
        denom = np.sqrt((g_test.var() + i_test[r].var()) / 2)
        out[f"dprime_{r}"] = float((g_test.mean() - i_test[r].mean()) / denom) if denom > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/results/boundary.json")
    args = ap.parse_args()

    configs = [(2, 2, 128), (2, 2, 256), (2, 2, 512),
               (2, 4, 128), (2, 4, 256),
               (1, 2, 128), (1, 4, 256),
               # disjoint windows at D=128: T*w = D
               (1, 1, 128), (2, 1, 128), (4, 1, 128),
               (1, 2, 64), (2, 2, 64), (4, 2, 64)]
    rows = []
    for B, w, T in configs:
        res = [run(s, 128, B, w, T, args.n) for s in args.seeds]
        row = {"B": float(B), "w": float(w), "T": float(T)}
        for k in res[0]:
            row[k] = float(np.mean([r[k] for r in res]))
            row[k + "_sd"] = float(np.std([r[k] for r in res]))
        rows.append(row)
        print(f"done B={B} w={w} T={T}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"genuine": GENUINE, "impostors": IMPOSTORS,
                                          "n": args.n, "seeds": args.seeds, "rows": rows}, indent=1))

    print("\n| B | w | T | t | genuine FNMR | impostor 0.80 FMR | impostor 0.82 FMR | d'(0.80) | d'(0.82) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {int(r['B'])} | {int(r['w'])} | {int(r['T'])} | {r['t']:.0f}"
              f" | {100*r['fnmr_test']:.1f}% | {100*r['fmr_0.8_test']:.2f}%"
              f" | {100*r['fmr_0.82_test']:.2f}% | {r['dprime_0.8']:.2f} | {r['dprime_0.82']:.2f} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
