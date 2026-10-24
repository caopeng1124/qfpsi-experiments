"""
Executable two-stage QFPSI: vectors -> tokens -> coarse/fine indexes ->
candidate filtering -> share reconstruction -> label output.

Every step is executed: real Ristretto255 OPRF, real Shamir shares over GF(2^8),
real packed indexes, real candidate-restricted lookup, and client-side
reconstruction from t_B distinct-window shares.

Measured per database mode (iid, clustered): latency, bytes, |Cand|, labels
returned, wrong labels, empty results, and whether the planted label was
recovered.

Run:  python experiments/proto/twostage_proto.py --n 10000 --q 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import statistics
import struct
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import qfpsi_sim as sim          # noqa: E402
import ristretto as rs           # noqa: E402

D = 128
T = 128
W_WIN = 2
B_FINE = 2
B_COARSE = 1
T_B = 61
T_C = 56
TAU = 0.85
TOKEN = 32
FRAME = 4

# ---------------------------------------------------------------- GF(2^8) ---
_EXP = np.zeros(512, dtype=np.uint8)
_LOG = np.zeros(256, dtype=np.int16)
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]
_MUL = np.zeros((256, 256), dtype=np.uint8)
for _a in range(1, 256):
    _MUL[_a] = _EXP[(_LOG[_a] + _LOG[:]) % 255]
_MUL[0, :] = 0
_MUL[:, 0] = 0


def share_matrix(labels: np.ndarray, t: int, T: int, rng) -> np.ndarray:
    n, L = labels.shape
    coeffs = rng.integers(0, 256, size=(n, L, t), dtype=np.uint8)
    coeffs[:, :, 0] = labels
    out = np.empty((n, T, L), dtype=np.uint8)
    for j in range(1, T + 1):
        acc = coeffs[:, :, t - 1].copy()
        for k in range(t - 2, -1, -1):
            acc = _MUL[j][acc] ^ coeffs[:, :, k]
        out[:, j - 1] = acc
    return out


def lagrange_weights(indices):
    w = {}
    for a in indices:
        num = np.uint8(1)
        den = np.uint8(1)
        for b in indices:
            if a == b:
                continue
            num = _MUL[num][b]
            den = _MUL[den][np.uint8(a ^ b)]
        w[a] = _MUL[num][_EXP[(255 - _LOG[den]) % 255]]
    return w


def reconstruct(shares, indices):
    w = lagrange_weights(indices)
    acc = np.zeros(16, dtype=np.uint8)
    for j, s in shares:
        acc ^= _MUL[w[j]][s]
    return acc.tobytes()


# ------------------------------------------------------------------ index ---
def sort_index(tags: np.ndarray, payload: np.ndarray):
    view = tags.view(np.uint64).reshape(-1, 4)
    order = np.lexsort((view[:, 1], view[:, 0]))
    return (np.ascontiguousarray(tags[order]),
            np.ascontiguousarray(view[order][:, 0]),
            np.ascontiguousarray(payload[order]))


def lookup(idx, tag: bytes):
    tags, prefix, payload = idx
    key = np.frombuffer(tag[:8], dtype=np.uint64)[0]
    lo = np.searchsorted(prefix, key, side="left")
    hi = np.searchsorted(prefix, key, side="right")
    return [payload[p].tobytes() for p in range(lo, hi) if tags[p].tobytes() == tag]


# ------------------------------------------------------- parallel indexing ---
def _worker(job):
    k, tok1, tok2, shares, lo = job
    n = tok1.shape[0]
    tags1 = np.empty((n * T, 32), dtype=np.uint8)
    tags2 = np.empty((n * T, 32), dtype=np.uint8)
    pay2 = np.empty((n * T, 21), dtype=np.uint8)
    for i in range(n):
        rec = lo + i
        for j in range(T):
            b1 = struct.pack("<QI", int(tok1[i, j]), j) + b"\x00" * 20
            b2 = struct.pack("<QI", int(tok2[i, j]), j) + b"\x00" * 20
            p = i * T + j                      # local index inside this chunk
            t1 = rs.tag(b1, rs.scalarmult(k, rs.h1(b1)))
            t2 = rs.tag(b2, rs.scalarmult(k, rs.h1(b2)))
            tags1[p] = np.frombuffer(t1, dtype=np.uint8)
            tags2[p] = np.frombuffer(t2, dtype=np.uint8)
            mask = hashlib.blake2b(b"MASK" + t2, digest_size=16).digest()
            pay2[p, :5] = np.frombuffer(struct.pack("<IB", rec, j), dtype=np.uint8)
            pay2[p, 5:] = shares[i, j] ^ np.frombuffer(mask, dtype=np.uint8)
    return tags1, tags2, pay2, lo


def build_two_stage(k, tok1, tok2, shares, workers):
    n = tok1.shape[0]
    bounds = np.linspace(0, n, workers + 1).astype(int)
    jobs = [(k, tok1[bounds[i]:bounds[i + 1]], tok2[bounds[i]:bounds[i + 1]],
             shares[bounds[i]:bounds[i + 1]], int(bounds[i])) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_worker, jobs))
    parts.sort(key=lambda x: x[3])
    tags1 = np.concatenate([p[0] for p in parts])
    tags2 = np.concatenate([p[1] for p in parts])
    pay2 = np.concatenate([p[2] for p in parts])
    handles = np.array([struct.pack("<IB", i // T, i % T) for i in range(n * T)],
                       dtype="S5").view(np.uint8).reshape(-1, 5)
    return sort_index(tags1, handles), sort_index(tags2, pay2)


# ----------------------------------------------------------------- server ---
class Server(threading.Thread):
    def __init__(self, k, coarse, fine, sock, stats):
        super().__init__(daemon=True)
        self.k, self.coarse, self.fine, self.sock, self.stats = k, coarse, fine, sock, stats

    def _recv(self, conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

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
                        # The literal protocol returns every (i,j) handle; that is
                        # O(T*N/2^w) bytes per query and unusable at small w. The
                        # server therefore counts per record (it already sees the
                        # hit pattern) and returns only records above t_c.
                        counts = {}
                        fanout = 0
                        for i in range(0, len(pts), TOKEN):
                            for h in lookup(self.coarse, pts[i:i + TOKEN]):
                                rec = struct.unpack("<I", h[:4])[0]
                                counts[rec] = counts.get(rec, 0) + 1
                                fanout += 1
                        cand = [rec for rec, c in counts.items() if c >= T_C]
                        out = np.asarray(cand, dtype=np.int32).tobytes()
                        conn.sendall(struct.pack("<I", len(out)) + out)
                        self.stats["coarse_hits"].append(fanout)
                        self.stats["cand"].append(len(cand))
                    elif mode == b"F":
                        n_cand = struct.unpack("<I", pts[:4])[0]
                        cand = set(np.frombuffer(pts[4:4 + 4 * n_cand], dtype=np.int32).tolist())
                        tags = pts[4 + 4 * n_cand:]
                        out = []
                        for i in range(0, len(tags), TOKEN):
                            out.extend(p for p in lookup(self.fine, tags[i:i + TOKEN])
                                       if struct.unpack("<I", p[:4])[0] in cand)
                        payload = b"".join(out)
                        conn.sendall(struct.pack("<I", len(payload)) + payload)
                        self.stats["fine_hits"].append(len(out))
                    else:
                        return
            finally:
                conn.close()


# ----------------------------------------------------------------- client ---
def tags_of(conn, tokens, counter):
    blinded, rs_ = [], []
    for t in tokens:
        r = rs.scalar_random()
        blinded.append(rs.scalarmult(r, rs.h1(t)))
        rs_.append(r)
    frame = b"E" + b"".join(blinded)
    conn.sendall(struct.pack("<I", len(frame)) + frame)
    counter[0] += FRAME + len(frame)
    n = struct.unpack("<I", conn.recv(FRAME))[0]
    ev = b""
    while len(ev) < n:
        ev += conn.recv(n - len(ev))
    counter[1] += FRAME + n
    out = []
    for i, (t, r) in enumerate(zip(tokens, rs_)):
        u = rs.scalarmult(rs.scalar_invert(r), ev[i * TOKEN:(i + 1) * TOKEN])
        out.append(rs.tag(t, u))
    return out


def client_query(addr, tokens_c, tokens_f, target, truth, stats):
    conn = socket.create_connection(addr)
    counter = [0, 0]

    import os
    dbg = os.environ.get("QFPSI_DEBUG")
    _t = time.perf_counter()
    c_tags = tags_of(conn, tokens_c, counter)
    if dbg: print(f"  [dbg] coarse OPRF {(time.perf_counter()-_t)*1e3:.1f} ms")
    frame = b"C" + b"".join(c_tags)
    conn.sendall(struct.pack("<I", len(frame)) + frame)
    counter[0] += FRAME + len(frame)
    n = struct.unpack("<I", conn.recv(FRAME))[0]
    raw = b""
    while len(raw) < n:
        raw += conn.recv(n - len(raw))
    counter[1] += FRAME + n

    cand = np.frombuffer(raw, dtype=np.int32).tolist()

    _t = time.perf_counter()
    f_tags = tags_of(conn, tokens_f, counter)
    if dbg: print(f"  [dbg] fine OPRF {(time.perf_counter()-_t)*1e3:.1f} ms")
    frame = (b"F" + struct.pack("<I", len(cand))
             + np.asarray(cand, dtype=np.int32).tobytes() + b"".join(f_tags))
    conn.sendall(struct.pack("<I", len(frame)) + frame)
    counter[0] += FRAME + len(frame)
    n = struct.unpack("<I", conn.recv(FRAME))[0]
    raw = b""
    while len(raw) < n:
        raw += conn.recv(n - len(raw))
    counter[1] += FRAME + n
    conn.close()

    if dbg: print(f"  [dbg] fine lookup {(time.perf_counter()-_t)*1e3:.1f} ms, bytes {len(raw)}")
    _t = time.perf_counter()
    shares = {}
    for k in range(0, len(raw), 21):
        i, j = struct.unpack("<IB", raw[k:k + 5])
        mask = hashlib.blake2b(b"MASK" + f_tags[j], digest_size=16).digest()
        sh = bytes(a ^ b for a, b in zip(raw[k + 5:k + 21], mask))
        shares.setdefault(i, []).append((j + 1, np.frombuffer(sh, dtype=np.uint8)))

    labels = []
    for i, sh in shares.items():
        if len({j for j, _ in sh}) >= T_B:
            labels.append((i, reconstruct(sh, [j for j, _ in sh])))

    if dbg: print(f"  [dbg] unmask+reconstruct {(time.perf_counter()-_t)*1e3:.1f} ms")
    stats["sent"].append(counter[0])
    stats["recv"].append(counter[1])
    stats["cand"].append(len(cand))
    stats["labels"].append(len(labels))
    stats["wrong"].append(sum(1 for i, lab in labels if lab != truth[i]))       # integrity
    stats["false_pos"].append(sum(1 for i, _ in labels if i != target))       # accuracy
    stats["empty"].append(0 if labels else 1)
    stats["recovered"].append(1 if any(i == target and lab == truth[i] for i, lab in labels) else 0)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--q", type=int, default=40)
    ap.add_argument("--near", type=int, default=50)
    ap.add_argument("--mode", choices=["iid", "clustered", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="experiments/results/twostage_proto.json")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    P = sim.random_rotation(rng, D)
    sigma = 1 / np.sqrt(D)
    zmin, zmax = -3 * sigma, 3 * sigma
    W = sim.windows(rng, D, T, W_WIN)

    qs = rng.normal(size=(args.q, D))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    db_iid = rng.normal(size=(args.n, D))
    db_iid /= np.linalg.norm(db_iid, axis=1, keepdims=True)

    def partner(qvec, rho):
        u = rng.normal(size=D)
        u -= (u * qvec).sum() * qvec
        u /= np.linalg.norm(u)
        return rho * qvec + np.sqrt(1 - rho**2) * u

    for i in range(args.q):
        db_iid[i] = partner(qs[i], TAU)

    truth = {i: hashlib.sha256(f"label{i}".encode()).digest()[:16] for i in range(args.n)}
    labels = np.array([np.frombuffer(truth[i], dtype=np.uint8) for i in range(args.n)])

    t0 = time.perf_counter()
    shares = share_matrix(labels, T_B, T, rng)
    t_shares = time.perf_counter() - t0
    print(f"shares: {args.n}x{T} over GF(2^8), t={T_B}: {t_shares:.1f}s")

    k = rs.scalar_random()

    results = {}
    modes = ["iid", "clustered"] if args.mode == "both" else [args.mode]
    for mode in modes:
        db = db_iid.copy()
        if mode == "clustered":
            for i in range(args.q):
                for j in range(args.near):
                    db[args.q + i * args.near + j] = partner(qs[i], rng.uniform(0.6, 0.8))

        t0 = time.perf_counter()
        U1 = sim.quantize(sim.encode(db, P, D, zmin, zmax), B_COARSE, zmin, zmax)
        U2 = sim.quantize(sim.encode(db, P, D, zmin, zmax), B_FINE, zmin, zmax)
        tok1 = sim.tokenize(U1, W, B_COARSE)
        tok2 = sim.tokenize(U2, W, B_FINE)
        coarse, fine = build_two_stage(k, tok1, tok2, shares, args.workers)
        t_build = time.perf_counter() - t0
        size_mib = (coarse[0].nbytes + coarse[2].nbytes + fine[0].nbytes + fine[2].nbytes) / 2**20
        print(f"[{mode}] two indexes: {t_build:.1f}s  {size_mib:.0f} MiB")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        addr = sock.getsockname()
        srv_stats = {"coarse_hits": [], "fine_hits": [], "cand": []}
        srv = Server(k, coarse, fine, sock, srv_stats)
        srv.start()

        U1q = sim.quantize(sim.encode(qs, P, D, zmin, zmax), B_COARSE, zmin, zmax)
        U2q = sim.quantize(sim.encode(qs, P, D, zmin, zmax), B_FINE, zmin, zmax)
        tok1q = sim.tokenize(U1q, W, B_COARSE)
        tok2q = sim.tokenize(U2q, W, B_FINE)

        stats = {x: [] for x in ("sent", "recv", "cand", "labels", "wrong",
                          "false_pos", "empty", "recovered")}
        lat = []
        for qi in range(args.q):
            tc_ = [struct.pack("<QI", int(v), j) + b"\x00" * 20
                   for j, v in enumerate(tok1q[qi])]
            tf_ = [struct.pack("<QI", int(v), j) + b"\x00" * 20
                   for j, v in enumerate(tok2q[qi])]
            t = time.perf_counter()
            client_query(addr, tc_, tf_, qi, truth, stats)
            lat.append((time.perf_counter() - t) * 1e3)
        lat.sort()
        r = {
            "queries": args.q, "build_s": t_build, "size_mib": size_mib,
            "latency_median_ms": statistics.median(lat),
            "latency_p95_ms": lat[int(0.95 * len(lat))],
            "cand_mean": statistics.fmean(stats["cand"]),
            "labels_mean": statistics.fmean(stats["labels"]),
            "wrong_mean": statistics.fmean(stats["wrong"]),
            "false_pos_mean": statistics.fmean(stats["false_pos"]),
            "empty_rate": statistics.fmean(stats["empty"]),
            "planted_recovered": statistics.fmean(stats["recovered"]),
            "bytes_sent": statistics.median(stats["sent"]),
            "bytes_recv": statistics.median(stats["recv"]),
            "coarse_fanout": statistics.median(srv_stats["coarse_hits"]),
        }
        results[mode] = r
        print(f"[{mode}] median {r['latency_median_ms']:.1f} ms  p95 {r['latency_p95_ms']:.1f} ms  "
              f"cand {r['cand_mean']:.1f}  labels {r['labels_mean']:.2f}  wrong {r['wrong_mean']:.2f}  "
              f"empty {100*r['empty_rate']:.0f}%  recovered {100*r['planted_recovered']:.0f}%  "
              f"bytes {r['bytes_sent']:.0f}+{r['bytes_recv']:.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n": args.n, "q": args.q, "T": T, "D": D, "t_c": T_C, "t_B": T_B,
        "shares_s": t_shares, "results": results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
