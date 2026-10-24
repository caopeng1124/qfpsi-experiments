"""
Correctness + microbenchmark for the 2HashDH OPRF over Ristretto255.

Checks that client and server computes agree, that the server's direct
evaluation (used for offline index construction) matches the blinded evaluation,
and measures per-scalar-multiplication and per-token costs.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ristretto as rs  # noqa: E402


def client_blind(token: bytes):
    r = rs.scalar_random()
    blinded = rs.scalarmult(r, rs.h1(token))
    return blinded, r


def server_eval(k: bytes, blinded: bytes) -> bytes:
    return rs.scalarmult(k, blinded)


def client_finalize(token: bytes, r: bytes, evaluated: bytes) -> bytes:
    unblinded = rs.scalarmult(rs.scalar_invert(r), evaluated)
    return rs.tag(token, unblinded)


def server_direct(k: bytes, token: bytes) -> bytes:
    """Offline index construction: server evaluates without blinding."""
    return rs.tag(token, rs.scalarmult(k, rs.h1(token)))


def bench(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)   # ms
    ts.sort()
    return {"median_ms": statistics.median(ts), "mean_ms": statistics.fmean(ts),
            "p95_ms": ts[int(0.95 * len(ts))], "n": n}


def main():
    out = {"libsodium": "1.0.22 / ristretto255"}
    k = rs.scalar_random()
    token = b"token-0123456789"

    # --- correctness ------------------------------------------------------
    blinded, r = client_blind(token)
    tag_client = client_finalize(token, r, server_eval(k, blinded))
    tag_direct = server_direct(k, token)
    out["correct"] = tag_client == tag_direct
    out["tag_len"] = len(tag_client)
    out["point_valid"] = rs.is_valid_point(blinded)
    if not out["correct"]:
        raise SystemExit("OPRF mismatch: client and server disagrees")
    print(f"correctness: {out['correct']}  tag={tag_client.hex()[:16]}…  "
          f"valid_point={out['point_valid']}")

    # --- microbenchmarks --------------------------------------------------
    h = rs.h1(token)
    rr = rs.scalar_random()
    out["scalarmult"] = bench(lambda: rs.scalarmult(rr, h), 20000)
    out["h1_from_hash"] = bench(lambda: rs.h1(token), 20000)
    out["blind"] = bench(lambda: client_blind(token), 5000)
    out["server_eval"] = bench(lambda: server_eval(k, blinded), 20000)
    out["server_direct"] = bench(lambda: server_direct(k, token), 5000)

    def full_token():
        b, r_ = client_blind(token)
        e = server_eval(k, b)
        return client_finalize(token, r_, e)

    out["online_token"] = bench(full_token, 3000)      # 3 mults + 2 hashes

    for name in ("scalarmult", "h1_from_hash", "blind", "server_eval",
                 "server_direct", "online_token"):
        b = out[name]
        print(f"{name:14s} median {b['median_ms']*1000:8.1f} us   "
              f"mean {b['mean_ms']*1000:8.1f} us   p95 {b['p95_ms']*1000:8.1f} us")

    # derived per-query costs (T_q tags)
    Tq = 128
    out["per_query"] = {
        "Tq": Tq,
        "client_ms": 2 * Tq * out["scalarmult"]["median_ms"],
        "server_ms": 1 * Tq * out["scalarmult"]["median_ms"],
        "measured_online_token_ms": Tq * out["online_token"]["median_ms"],
    }
    print(f"\nT_q={Tq}: client 2x{Tq} mults = {out['per_query']['client_ms']:.1f} ms, "
          f"server 1x{Tq} = {out['per_query']['server_ms']:.1f} ms, "
          f"measured per-token path = {out['per_query']['measured_online_token_ms']:.1f} ms")

    out["offline"] = {
        "per_token_ms": out["server_direct"]["median_ms"],
        "N1e4_T128_s": 1e4 * Tq * out["server_direct"]["median_ms"] / 1000,
        "N1e5_T128_s": 1e5 * Tq * out["server_direct"]["median_ms"] / 1000,
    }
    print(f"offline index build extrapolated: N=1e4 -> {out['offline']['N1e4_T128_s']:.0f} s, "
          f"N=1e5 -> {out['offline']['N1e5_T128_s']:.0f} s (single thread)")

    Path("experiments/results/oprf_bench.json").parent.mkdir(parents=True, exist_ok=True)
    Path("experiments/results/oprf_bench.json").write_text(json.dumps(out, indent=1))
    print("\nwrote experiments/results/oprf_bench.json")


if __name__ == "__main__":
    main()
