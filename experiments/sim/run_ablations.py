"""
Basic ablations of the calibrated predicate, all measured (no projections):

  E1  boundary expansion   r = 0 vs r = 1   -> FNMR gain vs transmitted-tag overhead
  E2  token count          T in {64,128,256} -> operating point and seed variance
  E3  dimensionality       D in {96,128,512,960} -> is the operating point D-invariant?

Protocol: for every (config, seed) we draw a calibration pair set and an
independent test pair set at the same similarity levels, calibrate t on the
calibration set at a target FMR, and report FMR/FNMR on the untouched test set.

Boundary expansion follows Section 4.3 of the paper: only the client expands,
only windows that contain a coordinate within tau_b of a bin edge are expanded,
each expanded window contributes one alternative per ambiguous coordinate, and
a window counts at most once. Transmitted tags T_q = T + sum_j a_j.
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
CLOSE_IDX = [RHO.index(r) for r in (0.85, 0.9, 0.95)]
FAR_IDX = list(range(RHO.index(0.5) + 1))
SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)


def pack_tokens(codes, W, B, alt_pos=None, alt_val=None):
    """Pack window codes into one integer and mix; optionally one coordinate replaced."""
    tok = np.zeros((codes.shape[0], W.shape[0]), dtype=np.uint64)
    for k in range(W.shape[1]):
        col = codes[:, W[:, k]].astype(np.uint64)
        if alt_pos is not None and k == alt_pos:
            col = alt_val
        tok |= col << np.uint64(B * k)
    return sim._mix64(tok)


def overlap_one_sided(tok_q, tok_x):
    return (tok_q == tok_x).sum(1)


def gen_pairs(rng, n, d):
    """Interleaved calibration/test pair arrays."""
    qs, xs = [], []
    for rho in RHO:
        q, x = sim.make_pairs(rng, rho, n, d)
        qs.append(q)
        xs.append(x)
    q = np.vstack(qs)
    x = np.vstack(xs)
    order = np.arange(0, q.shape[0])  # grouped by rho, aligned by index
    return q[order], x[order]


def run_one(seed, d, T, B, w, r, tau_b, n):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    sigma = 1.0 / np.sqrt(d)
    zmin, zmax = -3 * sigma, 3 * sigma

    qa, xa = gen_pairs(rng, n, d)   # calibration
    qb, xb = gen_pairs(rng, n, d)   # test

    def prep(q, x):
        phi_q = sim.positions(sim.encode(q, P, d, zmin, zmax), B, zmin, zmax)
        phi_x = sim.positions(sim.encode(x, P, d, zmin, zmax), B, zmin, zmax)
        return phi_q, phi_x

    phi_qa, phi_xa = prep(qa, xa)
    phi_qb, phi_xb = prep(qb, xb)
    W = sim.windows(rng, d, T, w)

    def codes_and_alts(phi, B):
        c = np.clip(np.rint(phi), 0, (1 << B) - 1).astype(np.uint8)
        return c

    c_qa, c_xa = codes_and_alts(phi_qa, B), codes_and_alts(phi_xa, B)
    c_qb, c_xb = codes_and_alts(phi_qb, B), codes_and_alts(phi_xb, B)

    # --- server side: centre tokens only (expansion is client-side) ----------
    tok_xa = sim.tokenize(c_xa, W, B)
    tok_xb = sim.tokenize(c_xb, W, B)

    def client_overlaps(phi_q, c_q, tok_x, expand):
        tok_c = sim.tokenize(c_q, W, B)
        ov = overlap_one_sided(tok_c, tok_x)
        tq = np.full(c_q.shape[0], T, dtype=np.float64)
        if not expand:
            return ov, tq
        amb = sim.ambiguous_mask(phi_q, tau_b, B)
        amb_w = amb[:, W]                                   # (n, T, w)
        a_j = amb_w.sum(2)                                  # ambiguous coords per window
        tq = T + a_j.sum(1)
        hit = (tok_c == tok_x)
        for k in range(w):
            sel = amb_w[:, :, k]
            col = c_q[:, W[:, k]]
            frac = phi_q[:, W[:, k]] - np.floor(phi_q[:, W[:, k]])
            # the centre code is floor(phi) when frac < 0.5, ceil(phi) otherwise;
            # the alternative is the other admissible code
            alt = np.where(frac < 0.5,
                           col.astype(np.int16) + 1,
                           col.astype(np.int16) - 1)
            alt = np.clip(alt, 0, (1 << B) - 1).astype(np.uint64)
            tok_alt = pack_tokens(c_q, W, B, alt_pos=k, alt_val=alt)
            hit |= (tok_alt == tok_x) & sel
        ov = hit.sum(1)
        return ov, tq

    ov_ca, tq_ca = client_overlaps(phi_qa, c_qa, tok_xa, r > 0)
    ov_cb, tq_cb = client_overlaps(phi_qb, c_qb, tok_xb, r > 0)

    def close_far(ov):
        ov = ov.reshape(len(RHO), n)
        close = np.concatenate([ov[i] for i in CLOSE_IDX]).astype(float)
        far = np.concatenate([ov[i] for i in FAR_IDX]).astype(float)
        return close, far

    cl_ca, fa_ca = close_far(ov_ca)
    cl_cb, fa_cb = close_far(ov_cb)

    out = {}
    for eps in (1e-2, 1e-3):
        op = sim.operating_point(cl_ca, fa_ca, T, eps)   # calibrate on calibration set
        if op is None:
            out[f"{eps:g}"] = None
            continue
        t = op[0]
        out[f"{eps:g}"] = {
            "t": int(t),
            "fmr_test": float((fa_cb >= t).mean()),
            "fnmr_test": float((cl_cb < t).mean()),
            "close_test_mean": float(cl_cb.mean()),
            "far_test_mean": float(fa_cb.mean()),
            "tq_mean": float(tq_cb.mean()),
            "tq_ratio": float(tq_cb.mean() / T),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="pairs per rho per set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/results/ablations.json")
    args = ap.parse_args()

    configs = []
    # E1: expansion at the default configuration
    configs += [("E1", dict(d=128, T=128, B=2, w=2, r=0, tau_b=0.15)),
                ("E1", dict(d=128, T=128, B=2, w=2, r=1, tau_b=0.15))]
    # E2: token count
    for T in (64, 128, 256):
        configs.append(("E2", dict(d=128, T=T, B=2, w=2, r=0, tau_b=0.15)))
    # E3: dimensionality
    for d in (96, 128, 512, 960):
        configs.append(("E3", dict(d=d, T=128, B=2, w=2, r=0, tau_b=0.15)))

    rows = []
    for group, cfg in configs:
        per_seed = [run_one(s, cfg["d"], cfg["T"], cfg["B"], cfg["w"], cfg["r"],
                            cfg["tau_b"], args.n) for s in args.seeds]
        for eps in ("0.01", "0.001"):
            vals = [p[eps] for p in per_seed if p.get(eps)]
            if not vals:
                continue
            rows.append({
                "group": group, "eps": float(eps), **cfg,
                "t": float(np.mean([v["t"] for v in vals])),
                "fmr": float(np.mean([v["fmr_test"] for v in vals])),
                "fmr_sd": float(np.std([v["fmr_test"] for v in vals])),
                "fnmr": float(np.mean([v["fnmr_test"] for v in vals])),
                "fnmr_sd": float(np.std([v["fnmr_test"] for v in vals])),
                "tq_ratio": float(np.mean([v["tq_ratio"] for v in vals])),
            })
        print(f"done {group} d={cfg['d']} T={cfg['T']} r={cfg['r']}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_per_rho": args.n, "seeds": args.seeds,
                                          "rho": RHO, "rows": rows}, indent=1))

    hdr = ("| group | D | T | r | target FMR | t | FMR(test) | FNMR(test) | T_q/T |\n"
           "|---|---|---|---|---|---|---|---|---|")
    print(hdr)
    for r_ in rows:
        print(f"| {r_['group']} | {r_['d']} | {r_['T']} | {r_['r']} | {r_['eps']:g}"
              f" | {r_['t']:.0f} | {100*r_['fmr']:.2f}% | {100*r_['fnmr']:.2f}%"
              f" | {r_['tq_ratio']:.2f} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
