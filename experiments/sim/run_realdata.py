"""
M7: the accuracy study on real vectors (SIFT1M, D=128, from ANN-Benchmarks).

The Gaussian study fixes the mechanism; this script checks the same quantities on
real descriptors: overlap against the cosine similarity of the pair, the
operating point at a 1% false-match budget, and the near-miss resolution that
decides whether a released label means anything.

Pairs come from the official 100-NN ground truth (genuine near neighbours) plus
random database records (far pairs). Similarity is the cosine of the anchored,
rotated vectors, which is what the protocol's tokens are built from.

Run:  python experiments/sim/run_realdata.py --shard sift --queries 800
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/sift-128-euclidean.hdf5")
    ap.add_argument("--queries", type=int, default=800, help="test queries sampled")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="experiments/results/real_sift.json")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
        nb = f["neighbors"][:]
    print(f"loaded SIFT: train {train.shape}, test {test.shape}, gt {nb.shape}")

    rng = np.random.default_rng(0)
    cal_idx = rng.choice(len(train), 20000, replace=False)
    anchor = train[cal_idx].mean(0)

    qsel = rng.choice(len(test), min(args.queries, len(test)), replace=False)
    Q = test[qsel].astype(np.float64)
    N = train[nb[qsel].reshape(-1)].astype(np.float64)          # neighbours
    R = train[rng.choice(len(train), len(qsel), replace=False)].astype(np.float64)  # far

    def to_z(v, P):
        d = v - anchor
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d @ P

    rows = []
    for seed in args.seeds:
        P = sim.random_rotation(np.random.default_rng(seed), train.shape[1])
        zq = to_z(Q, P)
        zn = to_z(N, P)
        zr = to_z(R, P)
        zq_rep = np.repeat(zq, len(nb[0]), axis=0)               # each query next to its neighbours
        cos_n = (zq_rep * zn).sum(1)                             # query vs its neighbours
        cos_r = (zq * zr).sum(1)                                 # query vs a random record

        W = sim.windows(np.random.default_rng(seed), train.shape[1], args.T, args.w)
        sig = 1.0 / np.sqrt(train.shape[1])

        def overlap(a, b):
            out = []
            for chunk in range(0, len(a), 20000):
                za, zb = a[chunk:chunk + 20000], b[chunk:chunk + 20000]
                Ua = sim.quantize(za, args.B, -3 * sig, 3 * sig)
                Ub = sim.quantize(zb, args.B, -3 * sig, 3 * sig)
                ta = sim.tokenize(Ua, W, args.B)
                tb = sim.tokenize(Ub, W, args.B)
                out.append(sim.overlap(ta, tb))
            return np.concatenate(out).astype(float)

        ov_n = overlap(zq_rep, zn)
        ov_r = overlap(zq, zr)

        # calibrate tau on the neighbour-cosine distribution (median of the top-100)
        tau = float(np.median(cos_n))
        close = cos_n >= tau
        near = (cos_n >= tau - 0.05) & (cos_n < tau)      # near misses just below tau
        far = np.concatenate([cos_r, cos_n[cos_n < tau - 0.10]])
        ov_far = np.concatenate([ov_r, ov_n[cos_n < tau - 0.10]])

        t = next(t for t in range(args.T + 1)
                 if float((ov_far >= t).mean()) <= 1e-2)

        rows.append({
            "seed": seed, "tau": tau,
            "cos_close_mean": float(cos_n[close].mean()),
            "cos_far_mean": float(cos_r.mean()),
            "overlap_close": float(ov_n[close].mean()),
            "overlap_near": float(ov_n[near].mean()),
            "overlap_far": float(ov_far.mean()),
            "t_far1": int(t),
            "fmr_far": float((ov_far >= t).mean()),
            "fnmr_close": float((ov_n[close] < t).mean()),
            "release_near": float((ov_n[near] >= t).mean()),
            "dprime_far": float((ov_n[close].mean() - ov_far.mean())
                                / np.sqrt((ov_n[close].var() + ov_far.var()) / 2)),
            "dprime_near": float((ov_n[close].mean() - ov_n[near].mean())
                                 / np.sqrt((ov_n[close].var() + ov_n[near].var()) / 2)),
        })
        r = rows[-1]
        print(f"seed {seed}: tau={tau:.3f}  close {r['cos_close_mean']:.3f}  far {r['cos_far_mean']:.3f}  "
              f"ov close/near/far {r['overlap_close']:.1f}/{r['overlap_near']:.1f}/{r['overlap_far']:.1f}  "
              f"FNMR {100*r['fnmr_close']:.1f}%  release_near {100*r['release_near']:.1f}%  "
              f"d'(far) {r['dprime_far']:.2f}  d'(near) {r['dprime_near']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "SIFT1M (ANN-Benchmarks)",
                                          "D": int(train.shape[1]), "B": args.B,
                                          "w": args.w, "T": args.T,
                                          "queries": len(qsel), "rows": rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
