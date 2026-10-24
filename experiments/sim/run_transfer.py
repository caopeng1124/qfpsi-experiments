"""
Transfer curve of the calibrated predicate.

Calibrate t so that 99% of genuine pairs (cos = 0.85) are released, then measure
the release probability P(overlap >= t) as a function of the pair's cosine
similarity. This is the effective decision rule that the protocol implements,
and it shows how wide the threshold band is in cosine units.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402

RHO = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.95]


def run(seed, d, B, w, T, n, tau):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    sigma = 1.0 / np.sqrt(d)
    zmin, zmax = -3 * sigma, 3 * sigma
    W = sim.windows(rng, d, T, w)

    def ov(rho, count):
        q, x = sim.make_pairs(rng, rho, count, d)
        U = sim.quantize(sim.encode(np.vstack([q, x]), P, d, zmin, zmax), B, zmin, zmax)
        tok = sim.tokenize(U, W, B)
        return sim.overlap(tok[:count], tok[count:]).astype(float)

    g_cal = ov(tau, n)
    t = int(np.quantile(g_cal, 0.01))          # 99% genuine recall
    curve = {rho: float((ov(rho, n) >= t).mean()) for rho in RHO}
    return t, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/results/transfer.json")
    args = ap.parse_args()

    configs = [(2, 2, 128), (2, 2, 512), (2, 4, 256)]
    out = {"rho": RHO, "tau": 0.85, "configs": {}}
    for B, w, T in configs:
        res = [run(s, 128, B, w, T, args.n, 0.85) for s in args.seeds]
        curves = np.array([[r[1][rho] for rho in RHO] for r in res])
        key = f"B{B}_w{w}_T{T}"
        out["configs"][key] = {"mean": curves.mean(0).tolist(),
                               "std": curves.std(0).tolist(),
                               "t": float(np.mean([r[0] for r in res]))}
        print(f"{key}: t={np.mean([r[0] for r in res]):.0f}  "
              + "  ".join(f"{rho:.2f}:{100*curves.mean(0)[i]:.0f}%"
                          for i, rho in enumerate(RHO)), file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
