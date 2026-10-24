"""
QFPSI system prototype: index construction, online query path, byte accounting.

This is the first executable piece of the protocol stack (not the full
two-stage/adaptive implementation): Ristretto255 OPRF, a packed token index,
Shamir-free payload lookup, and a real client/server round trip over TCP.

Measured outputs (written to experiments/results/proto.json):
  * offline index construction wall time and packed index size, N in {1e4, 1e5}
  * online latency per query (median / p95 over QUERIES queries)
  * bytes per query in each direction, split into
      blinded request (32*T_q), OPRF evaluation (32*T_q), tags (32*T_q),
      payloads (21*|hit|), framing
  * hit count per query
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
import ristretto as rs  # noqa: E402

T = 128
TOKEN = 32
PAYLOAD = 21
FRAME = 4          # uint32 length prefix per message


SHAPING = {"bw_mbps": 0.0, "rtt_ms": 0.0}     # 0 disables the emulation


def tx_delay(nbytes: int) -> float:
    """Seconds to spend on the wire for nbytes, one way (transmit + propagation)."""
    sec = 0.0
    if SHAPING["bw_mbps"]:
        sec += nbytes * 8 / (SHAPING["bw_mbps"] * 1e6)
    if SHAPING["rtt_ms"]:
        sec += SHAPING["rtt_ms"] / 2000.0
    return sec


def token_bytes(rec: int, win: int) -> bytes:
    return hashlib.blake2b(b"TOK" + struct.pack("<II", rec, win), digest_size=TOKEN).digest()


def build_chunk(args):
    k, lo, hi = args
    tags = np.empty(((hi - lo) * T, 32), dtype=np.uint8)
    pos = 0
    for i in range(lo, hi):
        for j in range(T):
            t = token_bytes(i, j)
            tags[pos] = np.frombuffer(rs.tag(t, rs.scalarmult(k, rs.h1(t))), dtype=np.uint8)
            pos += 1
    return tags


def build_index(k: bytes, n: int, workers: int):
    t0 = time.perf_counter()
    bounds = np.linspace(0, n, workers + 1).astype(int)
    jobs = [(k, int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(build_chunk, jobs))
    tags = np.concatenate(parts, axis=0)
    t_tags = time.perf_counter() - t0

    # sort lexicographically on the first 8 bytes (a 64-bit prefix), keep payloads aligned
    t0 = time.perf_counter()
    view = tags.view(np.uint64).reshape(-1, 4)
    order = np.lexsort((view[:, 1], view[:, 0]))
    tags = np.ascontiguousarray(tags[order])
    prefix = np.ascontiguousarray(view[order][:, 0])
    payload = np.frombuffer(
        b"".join(struct.pack("<IB", e // T, e % T) + bytes(16) for e in range(len(order))),
        dtype=np.uint8).reshape(-1, PAYLOAD).copy()   # placeholder share bytes
    t_sort = time.perf_counter() - t0

    idx_bytes = tags.nbytes + prefix.nbytes + payload.nbytes
    return {"tags": tags, "prefix": prefix, "payload": payload,
            "build_s": t_tags, "sort_s": t_sort,
            "entries": tags.shape[0], "nbytes": int(idx_bytes),
            "packed_per_entry": int(idx_bytes / tags.shape[0])}


def lookup(idx, tag: bytes):
    key = np.frombuffer(tag[:8], dtype=np.uint64)[0]
    lo = np.searchsorted(idx["prefix"], key, side="left")
    hi = np.searchsorted(idx["prefix"], key, side="right")
    out = []
    for p in range(lo, hi):
        if idx["tags"][p].tobytes() == tag:
            out.append(idx["payload"][p].tobytes())
    return out


class Server(threading.Thread):
    """Serves one client connection at a time, looping over frames."""

    def __init__(self, k, idx, sock, stats):
        super().__init__(daemon=True)
        self.k, self.idx, self.sock, self.stats = k, idx, sock, stats

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
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    head = self._recv(conn, FRAME)
                    if head is None:
                        break
                    n = struct.unpack("<I", head)[0]
                    body = self._recv(conn, n) if n else b""
                    if body is None:
                        break
                    mode, pts = body[:1], body[1:]
                    if mode == b"E":
                        out = b"".join(rs.scalarmult(self.k, pts[i:i + TOKEN])
                                       for i in range(0, len(pts), TOKEN))
                        conn.sendall(struct.pack("<I", len(out)) + out)
                    elif mode == b"Q":
                        hits = []
                        for i in range(0, len(pts), TOKEN):
                            hits.extend(lookup(self.idx, pts[i:i + TOKEN]))
                        out = b"".join(hits)
                        conn.sendall(struct.pack("<I", len(out)) + out)
                        self.stats["hits"].append(len(hits))
                    else:
                        return
            finally:
                conn.close()


def client_query(addr, tokens, stats):
    s = socket.create_connection(addr)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sent = recv = 0

    # round 1: blind + evaluate
    blinded, rs_ = [], []
    for t in tokens:
        r = rs.scalar_random()
        blinded.append(rs.scalarmult(r, rs.h1(t)))
        rs_.append(r)
    req = b"E" + b"".join(blinded)
    s.sendall(struct.pack("<I", len(req)) + req)
    sent += FRAME + len(req)
    time.sleep(tx_delay(FRAME + len(req)))
    n = struct.unpack("<I", s.recv(FRAME))[0]
    ev = b""
    while len(ev) < n:
        ev += s.recv(n - len(ev))
    recv += FRAME + n
    time.sleep(tx_delay(FRAME + n))

    # round 2: unblind, derive tags, look up
    tags = b"".join(rs.tag(t, rs.scalarmult(rs.scalar_invert(r), ev[i * TOKEN:(i + 1) * TOKEN]))
                    for i, (t, r) in enumerate(zip(tokens, rs_)))
    req2 = b"Q" + tags
    s.sendall(struct.pack("<I", len(req2)) + req2)
    sent += FRAME + len(req2)
    time.sleep(tx_delay(FRAME + len(req2)))
    n = struct.unpack("<I", s.recv(FRAME))[0]
    pay = b""
    while len(pay) < n:
        pay += s.recv(n - len(pay))
    recv += FRAME + n
    time.sleep(tx_delay(FRAME + n))
    s.close()
    stats["sent"].append(sent)
    stats["recv"].append(recv)
    return pay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hits-per-query", type=int, default=80,
                    help="query tokens equal to one record's tokens (simulated overlap)")
    ap.add_argument("--shaping", choices=["none", "fast", "slow"], default="none",
                    help="fast = 500 MB/s + 0.5 ms RTT, slow = 40 MB/s + 34 ms RTT "
                         "(FLPSI settings), applied as a user-space emulation")
    ap.add_argument("--out", default="experiments/results/proto.json")
    args = ap.parse_args()

    if args.shaping == "fast":
        SHAPING.update(bw_mbps=500 * 8, rtt_ms=0.5)
    elif args.shaping == "slow":
        SHAPING.update(bw_mbps=40 * 8, rtt_ms=34.0)
    print(f"shaping: {args.shaping}  bw={SHAPING['bw_mbps']:.0f} Mbps  "
          f"rtt={SHAPING['rtt_ms']} ms")

    k = rs.scalar_random()
    print(f"building index: N={args.n}, T={T}, workers={args.workers}")
    idx = build_index(k, args.n, args.workers)
    print(f"  entries {idx['entries']:,}  build {idx['build_s']:.1f}s  "
          f"sort {idx['sort_s']:.1f}s  {idx['nbytes']/2**20:.0f} MiB "
          f"({idx['packed_per_entry']} B/entry)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    addr = sock.getsockname()
    stats = {"hits": [], "sent": [], "recv": []}
    srv = Server(k, idx, sock, stats)
    srv.start()

    # query tokens: hits-per-query tokens of record 0 plus random tokens
    base = [token_bytes(0, j) for j in range(args.hits_per_query)]
    rng = np.random.default_rng(0)
    extra = [rng.bytes(TOKEN) for _ in range(T - args.hits_per_query)]
    tokens = base + extra

    lat = []
    for _ in range(5):                      # warm-up
        client_query(addr, tokens, {"hits": [], "sent": [], "recv": []})
    for _ in range(args.queries):
        t0 = time.perf_counter()
        client_query(addr, tokens, stats)
        lat.append((time.perf_counter() - t0) * 1e3)

    lat.sort()
    res = {
        "n": args.n, "T": T, "queries": args.queries, "workers": args.workers,
        "shaping": {"mode": args.shaping, **SHAPING},
        "index": {"entries": idx["entries"], "build_s": idx["build_s"],
                  "sort_s": idx["sort_s"], "bytes": idx["nbytes"],
                  "packed_per_entry": idx["packed_per_entry"]},
        "latency_ms": {"median": statistics.median(lat), "mean": statistics.fmean(lat),
                       "p95": lat[int(0.95 * len(lat))], "min": lat[0], "max": lat[-1]},
        "bytes": {"sent_median": statistics.median(stats["sent"]),
                  "recv_median": statistics.median(stats["recv"]),
                  "total_median": statistics.median(stats["sent"]) + statistics.median(stats["recv"])},
        "hits_median": statistics.median(stats["hits"]),
        "components": {
            "oprf_request": TOKEN * T + FRAME,
            "oprf_evaluation": TOKEN * T + FRAME,
            "tag_request": TOKEN * T + FRAME,
            "payloads_per_hit": PAYLOAD,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))

    b = res["bytes"]
    print(f"\nlatency over {args.queries} queries: median {res['latency_ms']['median']:.2f} ms"
          f"  p95 {res['latency_ms']['p95']:.2f} ms")
    print(f"bytes/query: sent {b['sent_median']:.0f}  recv {b['recv_median']:.0f}"
          f"  total {b['total_median']:.0f}  hits {res['hits_median']:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
