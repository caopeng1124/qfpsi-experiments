"""
Database-level measurement of the hierarchical two-stage protocol (Section 5.4).

Builds a controlled database of N = 10,000 records with:
  * one planted match per query at cosine 0.85 (the semantic positive), and
  * either (a) i.i.d. records, or (b) 20% near-miss records at cosine 0.6-0.8,
which is the regime where a false label can actually be released.

Measures, per coarse threshold t_c:
  |Cand|/N            candidate ratio (what the earlier draft asserted as 0.19%)
  planted-in-Cand     fraction of queries whose true match survives the coarse stage
  labels / wrong      labels returned and wrong labels per query after fine matching
  eta_hit             fraction of coarse query tokens that hit the index

The fine stage uses B=2, w=2, T=128 with t_2 taken from the calibrated
operating point in Table 2 (t=61 at FMR<=1%), restricted to the candidate set.
Both stages are those of the paper; nothing is projected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfpsi_sim as sim  # noqa: E402

T = 128
W_WIN = 2
B_FINE = 2
T_FINE = 61          # calibrated in Table 2 at FMR <= 1%
TAU = 0.85


def _pair(rng, q, rho, d):
    """Vectors x with <q, x> = rho (q is a unit vector)."""
    u = rng.normal(size=d)
    u -= (u * q).sum() * q
    u /= np.linalg.norm(u)
    return rho * q + np.sqrt(1 - rho**2) * u


def build_db(rng, n, q, d, mode):
    """Database layout: rows [0,q) planted matches, then (clustered only)
    q slabs of K near-miss records, the rest i.i.d."""
    db = rng.normal(size=(n, d))
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    qs = rng.normal(size=(q, d))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    for i in range(q):
        db[i] = _pair(rng, qs[i], TAU, d)
    if mode == "clustered":
        K = 50
        for i in range(q):
            lo = q + i * K
            for j in range(K):
                db[lo + j] = _pair(rng, qs[i], rng.uniform(0.6, 0.8), d)
    elif mode != "iid":
        raise ValueError(mode)
    return db, qs


def run(seed, n, q, d, mode, tc_grid):
    rng = np.random.default_rng(seed)
    P = sim.random_rotation(rng, d)
    sigma = 1.0 / np.sqrt(d)
    zmin, zmax = -3 * sigma, 3 * sigma
    W = sim.windows(rng, d, T, W_WIN)

    db, qs = build_db(rng, n, q, d, mode)
    U1 = sim.quantize(sim.encode(db, P, d, zmin, zmax), 1, zmin, zmax)
    U1q = sim.quantize(sim.encode(qs, P, d, zmin, zmax), 1, zmin, zmax)
    U2 = sim.quantize(sim.encode(db, P, d, zmin, zmax), B_FINE, zmin, zmax)
    U2q = sim.quantize(sim.encode(qs, P, d, zmin, zmax), B_FINE, zmin, zmax)

    tok1_db = sim.tokenize(U1, W, 1)
    tok1_q = sim.tokenize(U1q, W, 1)
    tok2_db = sim.tokenize(U2, W, B_FINE)
    tok2_q = sim.tokenize(U2q, W, B_FINE)

    rows = []
    for tc in tc_grid:
        cands, planted_ok, eta, n_labels, n_wrong, empty = [], [], [], [], [], []
        for i in range(q):
            cc = (tok1_db == tok1_q[i]).sum(1)            # coarse count per record
            cand = np.where(cc >= tc)[0]
            cands.append(cand.size / n)
            planted_ok.append(cc[i] >= tc)
            eta.append((tok1_q[i][None, :] == tok1_db).any(0).mean())
            if cand.size == 0:
                n_labels.append(0); n_wrong.append(0); empty.append(True)
                continue
            fo = (tok2_db[cand] == tok2_q[i]).sum(1)       # fine overlap on candidates
            released = cand[fo >= T_FINE]
            n_labels.append(released.size)
            # a released record is wrong unless it is the planted match i
            n_wrong.append(int((released != i).sum()))
            empty.append(released.size == 0)
        rows.append({
            "mode": mode, "tc": int(tc),
            "cand_ratio": float(np.mean(cands)),
            "planted_in_cand": float(np.mean(planted_ok)),
            "eta_hit": float(np.mean(eta)),
            "labels_per_query": float(np.mean(n_labels)),
            "wrong_per_query": float(np.mean(n_wrong)),
            "empty_rate": float(np.mean(empty)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--q", type=int, default=100)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="experiments/results/twostage.json")
    args = ap.parse_args()
    tc_grid = [40, 48, 56, 64, 72, 80, 88, 96]

    all_rows = []
    for mode in ("iid", "clustered"):
        acc = {}
        for s in args.seeds:
            for row in run(s, args.n, args.q, args.d, mode, tc_grid):
                acc.setdefault(row["tc"], []).append(row)
            print(f"seed {s} {mode} done", file=sys.stderr)
        for tc, rows in acc.items():
            agg = {"mode": mode, "tc": int(tc)}
            for key in ("cand_ratio", "planted_in_cand", "eta_hit",
                        "labels_per_query", "wrong_per_query", "empty_rate"):
                agg[key] = float(np.mean([row[key] for row in rows]))
            all_rows.append(agg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"N": args.n, "Q": args.q, "D": args.d,
                                          "T": T, "w": W_WIN, "B_fine": B_FINE,
                                          "t_fine": T_FINE, "tau": TAU,
                                          "seeds": args.seeds, "rows": all_rows}, indent=1))

    print(f"\nN={args.n} Q={args.q} seeds={args.seeds}  fine: B={B_FINE}, w={W_WIN}, t={T_FINE}\n")
    print("| DB | t_c | |Cand|/N | planted in Cand | eta_hit | labels/q | wrong/q | empty |")
    print("|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        print(f"| {r['mode']} | {r['tc']} | {100*r['cand_ratio']:.3f}% |"
              f" {100*r['planted_in_cand']:.1f}% | {100*r['eta_hit']:.1f}% |"
              f" {r['labels_per_query']:.2f} | {r['wrong_per_query']:.2f} | {100*r['empty_rate']:.1f}% |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
