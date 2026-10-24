"""
Adaptive-B, executed: coarse stage -> feedback -> policy -> chosen fine index.

The paper leaves the policy unspecified ("its cutoffs are deployment-specific"),
so we calibrate it instead of hard-coding it:

  1. build the coarse index and one fine index per B in {1,2,4};
  2. on a calibration set of queries (different vectors from the test set),
     record the feedback vector (|Cand|, nu1, nu1-nu2, eta_hit) and whether
     each width recovers the planted label;
  3. fit the rule "use the finest width whose recovery rate is at least the
     target, as a function of nu1" -- a single threshold theta that is the
     smallest nu1 above which B=2 never fails on the calibration set;
  4. on held-out queries, run the policy and compare it with fixed B=1 and B=2.

Measured: chosen-width distribution, recovered rate, false positives, payload
bytes, latency.

Run:  python experiments/proto/adaptive_proto.py --n 5000 --q 30 --calib 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import qfpsi_sim as sim          # noqa: E402
import ristretto as rs           # noqa: E402
import twostage_proto as tp      # noqa: E402

D, T, W_WIN = tp.D, tp.T, tp.W_WIN
T_C = tp.T_C
TOKEN, FRAME = tp.TOKEN, tp.FRAME
T_B = {1: 72, 2: 61, 4: 9}       # calibrated thresholds (Table 2, FMR <= 1%)
WIDTHS = (1, 2, 4)


class MultiServer(tp.Server):
    """Server holding one fine index per width; the request names the width."""

    def __init__(self, k, coarse, fines, sock, stats):
        super().__init__(k, coarse, fines[WIDTHS[0]], sock, stats)
        self.fines = fines

    def run(self):
        while True:
            conn, _ = self.sock.accept()
            try:
                while True:
                    head = self._recv(conn, FRAME)
                    if head is None:
                        break
                    body = self._recv(conn, struct.unpack("<I", head)[0])
                    if body is None:
                        break
                    mode, pts = body[:1], body[1:]
                    if mode == b"E":
                        out = b"".join(rs.scalarmult(self.k, pts[i:i + TOKEN])
                                       for i in range(0, len(pts), TOKEN))
                        conn.sendall(struct.pack("<I", len(out)) + out)
                    elif mode == b"C":
                        counts, fanout = {}, 0
                        for i in range(0, len(pts), TOKEN):
                            for h in tp.lookup(self.coarse, pts[i:i + TOKEN]):
                                rec = struct.unpack("<I", h[:4])[0]
                                counts[rec] = counts.get(rec, 0) + 1
                                fanout += 1
                        keep = [r for r, c in counts.items() if c >= T_C]
                        out = b"".join(struct.pack("<iH", r, min(counts[r], 65535))
                                       for r in keep)
                        conn.sendall(struct.pack("<I", len(out)) + out)
                        self.stats["coarse_hits"].append(fanout)
                        self.stats["counts"].append(counts)
                    elif mode == b"F":
                        B = pts[0]
                        n_cand = struct.unpack("<I", pts[1:5])[0]
                        cand = set(np.frombuffer(pts[5:5 + 4 * n_cand], dtype=np.int32).tolist())
                        tags = pts[5 + 4 * n_cand:]
                        idx = self.fines[B]
                        out = []
                        for i in range(0, len(tags), TOKEN):
                            out.extend(p for p in tp.lookup(idx, tags[i:i + TOKEN])
                                       if struct.unpack("<I", p[:4])[0] in cand)
                        payload = b"".join(out)
                        conn.sendall(struct.pack("<I", len(payload)) + payload)
                        self.stats["fine_hits"].append(len(out))
                    else:
                        return
            finally:
                conn.close()


def query(addr, qvec, P, zmin, zmax, W, policy, truth, target, stats):
    """One query. policy: 'fixed1' | 'fixed2' | 'adaptive'."""
    conn = socket.create_connection(addr)
    sent = recv = 0

    def rpc(frame):
        nonlocal sent, recv
        conn.sendall(struct.pack("<I", len(frame)) + frame)
        sent += FRAME + len(frame)
        n = struct.unpack("<I", conn.recv(FRAME))[0]
        buf = b""
        while len(buf) < n:
            buf += conn.recv(n - len(buf))
        recv += FRAME + n
        return buf

    def tags(tokens):
        blinded, rs_ = [], []
        for t in tokens:
            r = rs.scalar_random()
            blinded.append(rs.scalarmult(r, rs.h1(t)))
            rs_.append(r)
        ev = rpc(b"E" + b"".join(blinded))
        out = []
        for i, (t, r) in enumerate(zip(tokens, rs_)):
            u = rs.scalarmult(rs.scalar_invert(r), ev[i * TOKEN:(i + 1) * TOKEN])
            out.append(rs.tag(t, u))
        return out

    toks = {}
    for B in WIDTHS:
        U = sim.quantize(sim.encode(qvec.reshape(1, -1), P, D, zmin, zmax), B, zmin, zmax)
        tk = sim.tokenize(U, W, B)[0]
        toks[B] = [struct.pack("<QI", int(v), j) + b"\x00" * 20 for j, v in enumerate(tk)]

    c_tags = tags(toks[1])
    raw = rpc(b"C" + b"".join(c_tags))
    counts = {}
    for i in range(0, len(raw), 6):
        rid, cnt = struct.unpack("<iH", raw[i:i + 6])
        counts[rid] = cnt
    cand = list(counts)

    # feedback vector: |Cand|, nu1, nu1-nu2, eta_hit  (eta not tracked server-side)
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    nu1 = top[0][1] if top else 0
    nu2 = top[1][1] if len(top) > 1 else 0
    feedback = {"cand": len(cand), "nu1": nu1, "gap": nu1 - nu2}

    if policy == "fixed1":
        B = 1
    elif policy == "fixed2":
        B = 2
    else:
        B = 2 if nu1 >= stats["theta"] else 1

    f_tags = tags(toks[B])
    frame = (b"F" + bytes([B]) + struct.pack("<I", len(cand))
             + np.asarray(cand, dtype=np.int32).tobytes() + b"".join(f_tags))
    raw = rpc(frame)
    conn.close()

    shares = {}
    for k in range(0, len(raw), 21):
        i, j = struct.unpack("<IB", raw[k:k + 5])
        mask = hashlib.blake2b(b"MASK" + f_tags[j], digest_size=16).digest()
        sh = bytes(a ^ b for a, b in zip(raw[k + 5:k + 21], mask))
        shares.setdefault(i, []).append((j + 1, np.frombuffer(sh, dtype=np.uint8)))

    labels = [(i, tp.reconstruct(sh, [j for j, _ in sh]))
              for i, sh in shares.items() if len({j for j, _ in sh}) >= T_B[B]]
    stats["B"].append(B)
    stats["sent"].append(sent)
    stats["recv"].append(recv)
    stats["labels"].append(len(labels))
    stats["false_pos"].append(sum(1 for i, _ in labels if i != target))
    stats["recovered"].append(1 if any(i == target and lab == truth[i] for i, lab in labels) else 0)
    stats.setdefault("released", []).append(1 if any(i == target for i, _ in labels) else 0)
    stats["feedback"].append(feedback)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--q", type=int, default=30)
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="experiments/results/adaptive_proto.json")
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    P = sim.random_rotation(rng, D)
    sg = 1 / np.sqrt(D)
    zmin, zmax = -3 * sg, 3 * sg
    W = sim.windows(rng, D, T, W_WIN)

    rhos = [0.90, 0.85, 0.82, 0.78]      # heterogeneous difficulty
    n_q = args.q + args.calib
    qs = rng.normal(size=(n_q, D))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    db = rng.normal(size=(args.n, D))
    db /= np.linalg.norm(db, axis=1, keepdims=True)

    def partner(qv, rho):
        u = rng.normal(size=D)
        u -= (u * qv).sum() * qv
        u /= np.linalg.norm(u)
        return rho * qv + np.sqrt(1 - rho**2) * u

    should = [rhos[i % len(rhos)] >= tp.TAU for i in range(n_q)]
    for i in range(n_q):
        db[i] = partner(qs[i], rhos[i % len(rhos)])
    truth = {i: hashlib.sha256(f"label{i}".encode()).digest()[:16] for i in range(args.n)}
    labels = np.array([np.frombuffer(truth[i], dtype=np.uint8) for i in range(args.n)])
    shares = tp.share_matrix(labels, tp.T_B, T, rng)

    k = rs.scalar_random()
    tok = {}
    for B in WIDTHS:
        U = sim.quantize(sim.encode(db, P, D, zmin, zmax), B, zmin, zmax)
        tok[B] = sim.tokenize(U, W, B)
    t0 = time.perf_counter()
    coarse, fine1 = tp.build_two_stage(k, tok[1], tok[1], shares, args.workers)
    _, fine2 = tp.build_two_stage(k, tok[1], tok[2], shares, args.workers)
    _, fine4 = tp.build_two_stage(k, tok[1], tok[4], shares, args.workers)
    build_s = time.perf_counter() - t0
    fines = {1: fine1, 2: fine2, 4: fine4}
    print(f"indexes (coarse + 3 widths): {build_s:.1f}s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    addr = sock.getsockname()
    srv_stats = {"coarse_hits": [], "fine_hits": [], "counts": []}
    srv = MultiServer(k, coarse, fines, sock, srv_stats)
    srv.start()

    # --- calibration: which B recovers, as a function of nu1 -----------------
    cal = []
    for qi in range(args.calib):
        st = {"theta": 0, **{x: [] for x in ("B", "sent", "recv", "labels",
                                            "false_pos", "recovered", "feedback")}}
        rec = {}
        for pol, B in (("fixed1", 1), ("fixed2", 2)):
            st2 = {x: [] for x in ("B", "sent", "recv", "labels", "false_pos",
                                   "recovered", "feedback")}
            st2["theta"] = 0
            query(addr, qs[qi], P, zmin, zmax, W, pol, truth, qi, st2)
            rec[B] = st2["recovered"][0]
            rec["nu1"] = st2["feedback"][0]["nu1"]
        cal.append(rec)

    ok2 = [c["nu1"] for c in cal if c[2]]
    bad2 = [c["nu1"] for c in cal if not c[2]]
    theta = int(max(bad2)) + 1 if bad2 else 0
    print(f"calibration: {len(cal)} queries, B=2 failures {len(bad2)}, theta={theta}")

    # --- evaluation ---------------------------------------------------------
    results = {}
    for pol in ("fixed1", "fixed2", "adaptive"):
        stats = {"theta": theta, **{x: [] for x in ("B", "sent", "recv", "labels",
                                                    "false_pos", "recovered", "released",
                                                    "feedback")}}
        lat = []
        for qi in range(args.calib, n_q):
            t = time.perf_counter()
            query(addr, qs[qi], P, zmin, zmax, W, pol, truth, qi, stats)
            lat.append((time.perf_counter() - t) * 1e3)
        lat.sort()
        rel = np.array(stats["released"], dtype=float)
        should_t = np.array(should[args.calib:], dtype=bool)
        results[pol] = {
            "queries": args.q,
            "latency_median_ms": statistics.median(lat),
            "tpr": float(rel[should_t].mean()) if should_t.any() else 0.0,
            "fpr": float(rel[~should_t].mean()) if (~should_t).any() else 0.0,
            "recovered": statistics.fmean(stats["recovered"]),
            "false_pos_mean": statistics.fmean(stats["false_pos"]),
            "labels_mean": statistics.fmean(stats["labels"]),
            "bytes_sent": statistics.median(stats["sent"]),
            "bytes_recv": statistics.median(stats["recv"]),
            "B_hist": {int(b): stats["B"].count(b) for b in sorted(set(stats["B"]))},
        }
        r = results[pol]
        print(f"{pol:9s} TPR {100*r['tpr']:5.1f}%  FPR {100*r['fpr']:5.1f}%  false-pos {r['false_pos_mean']:5.2f}  "
              f"latency {r['latency_median_ms']:6.1f} ms  bytes {r['bytes_sent']:.0f}+{r['bytes_recv']:.0f}  "
              f"B={r['B_hist']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n": args.n, "q": args.q, "calib": args.calib, "build_s": build_s,
        "theta": theta, "t_B": T_B, "results": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
