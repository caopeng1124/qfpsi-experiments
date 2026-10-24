"""
Controlled study of the calibrated token-overlap predicate, 5 rotation seeds.

For each rotation seed:
  * build pairs at controlled cosine similarity (D=128 by default)
  * for every (B, w) in the grid, measure the overlap distribution per rho
  * calibrate t_B,w on FMR<=eps and record the resulting FNMR
Aggregate mean +/- std across seeds, dump JSON, print a markdown table.

The output is what Section 6 of the paper reports: a *measured* characterisation
of the predicate on Gaussian pairs, with the calibration protocol applied.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402

RHO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
GRID = [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (4, 8)]
CLOSE = (0.85, 0.90, 0.95)
FAR_MAX = 0.5


def run_seed(seed, d, T, n_per_rho, eps_list):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    zmin, zmax = -3.0 / np.sqrt(d), 3.0 / np.sqrt(d)

    vecs = []
    for rho in RHO:
        q, x = sim.make_pairs(rng, rho, n_per_rho, d)
        vecs.append(np.stack([q, x], axis=1).reshape(-1, d))
    V = np.vstack(vecs)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Z = sim.encode(V, P, d, zmin, zmax)

    out = {}
    for B, w in GRID:
        U = sim.quantize(Z, B, zmin, zmax)
        W = sim.windows(rng, d, T, w)
        tok = sim.tokenize(U, W, B)
        iq = np.arange(0, V.shape[0], 2)
        ov = sim.overlap(tok[iq], tok[iq + 1]).reshape(len(RHO), n_per_rho)
        close = np.concatenate([ov[RHO.index(r)] for r in CLOSE]).astype(float)
        far = np.concatenate([ov[i] for i in range(RHO.index(FAR_MAX) + 1)]).astype(float)
        rec = {
            "mean": ov.mean(1).tolist(),
            "std": ov.std(1).tolist(),
            "close_mean": float(close.mean()),
            "close_std": float(close.std()),
            "far_mean": float(far.mean()),
            "far_std": float(far.std()),
            "dprime": float((close.mean() - far.mean())
                            / np.sqrt((close.var() + far.var()) / 2)),
            "ops": {},
        }
        for eps in eps_list:
            op = sim.operating_point(close, far, T, eps)
            if op:
                rec["ops"][f"{eps:g}"] = {"t": int(op[0]), "fmr": op[1], "fnmr": op[2]}
        out[f"B{B}w{w}"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--n", type=int, default=1000, help="pairs per similarity level")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/results/predicate_multiseed.json")
    args = ap.parse_args()
    eps_list = [1e-1, 1e-2, 1e-3]

    runs = []
    for s in args.seeds:
        runs.append(run_seed(s, args.d, args.T, args.n, eps_list))
        print(f"seed {s} done", file=sys.stderr)

    agg = {}
    for key in runs[0]:
        per = [r[key] for r in runs]
        a = {
            "mean": np.mean([p["mean"] for p in per], 0).tolist(),
            "mean_std": np.std([p["mean"] for p in per], 0).tolist(),
            "close_mean": float(np.mean([p["close_mean"] for p in per])),
            "close_seed_std": float(np.std([p["close_mean"] for p in per])),
            "far_mean": float(np.mean([p["far_mean"] for p in per])),
            "far_seed_std": float(np.std([p["far_mean"] for p in per])),
            "dprime": float(np.mean([p["dprime"] for p in per])),
            "dprime_seed_std": float(np.std([p["dprime"] for p in per])),
            "ops": {},
        }
        for eps in eps_list:
            k = f"{eps:g}"
            ops = [p["ops"][k] for p in per if k in p["ops"]]
            if ops:
                a["ops"][k] = {
                    "t": float(np.mean([o["t"] for o in ops])),
                    "fmr": float(np.mean([o["fmr"] for o in ops])),
                    "fmr_seed_std": float(np.std([o["fmr"] for o in ops])),
                    "fnmr": float(np.mean([o["fnmr"] for o in ops])),
                    "fnmr_seed_std": float(np.std([o["fnmr"] for o in ops])),
                }
        agg[key] = a

    payload = {"D": args.d, "T": args.T, "n_per_rho": args.n, "seeds": args.seeds,
               "rho": RHO, "close_rhos": CLOSE, "far_max": FAR_MAX, "aggregate": agg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))

    print(f"\nD={args.d} T={args.T} pairs/rho={args.n} seeds={args.seeds}\n")
    print("| B | w | overlap close | overlap far | d' | t @FMR<=1% | FNMR @FMR<=1% | t @FMR<=0.1% | FNMR @FMR<=0.1% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, a in agg.items():
        B = key[1:key.index("w")]
        w = key[key.index("w") + 1:]
        o1 = a["ops"].get("0.01", {})
        o2 = a["ops"].get("0.001", {})
        print(f"| {B} | {w} | {a['close_mean']:.1f}±{a['close_std'] if 'close_std' in a else a['close_seed_std']:.1f}"
              f" | {a['far_mean']:.1f}±{a['far_seed_std']:.1f} | {a['dprime']:.2f}±{a['dprime_seed_std']:.2f}"
              f" | {o1.get('t', float('nan')):.0f} | {100*o1.get('fnmr', float('nan')):.2f}\\%"
              f" | {o2.get('t', float('nan')):.0f} | {100*o2.get('fnmr', float('nan')):.2f}\\% |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
