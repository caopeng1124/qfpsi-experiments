"""
QFPSI simulation library (accuracy layer).

Implements the construction of Section 4 of the paper exactly:
  normalize -> random rotation -> B-bit uniform quantization
  -> T random sub-dimension windows of size w -> hash -> token overlap.

Only numpy is required. Used by run_multiseed.py and by future dataset runs.
"""
from __future__ import annotations

import numpy as np


def random_rotation(rng, d):
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def make_pairs(rng, rho, n, d):
    """Unit vectors (q, x) with <q,x> = rho exactly."""
    q = rng.normal(size=(n, d))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    u = rng.normal(size=(n, d))
    u -= (u * q).sum(1, keepdims=True) * q
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = rho * q + np.sqrt(max(0.0, 1.0 - rho * rho)) * u
    return q, x


def encode(v, P, d, zmin, zmax):
    """Normalize, rotate, and rescale to the quantization window."""
    z = v @ P
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return np.clip(z, zmin, zmax)


def positions(z, B, zmin, zmax):
    """Continuous quantization coordinate phi in [0, 2^B-1] (before rounding)."""
    lv = (1 << B) - 1
    return (z - zmin) / (zmax - zmin) * lv


def quantize_phi(phi):
    return np.clip(np.rint(phi), 0, np.inf).astype(np.uint8)


def ambiguous_mask(phi, tau_b, B):
    """Coordinates within tau_b of a bin edge (k+0.5) admit two codes."""
    frac = phi - np.floor(phi)
    near = np.abs(frac - 0.5) <= tau_b
    lv = (1 << B) - 1
    # an ambiguity at the two extreme bins has no neighbour outside the range
    valid = (np.floor(phi) >= 0) & (np.ceil(phi) <= lv)
    return near & valid


def quantize(z, B, zmin, zmax):
    lv = (1 << B) - 1
    u = np.rint((z - zmin) / (zmax - zmin) * lv)
    return np.clip(u, 0, lv).astype(np.uint8)


def windows(rng, d, T, w):
    return np.stack([rng.choice(d, w, replace=False) for _ in range(T)])


def _mix64(x):
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def tokenize(u, W, B):
    """Pack each window's w codes into one integer, then mix (stand-in for H)."""
    tok = np.zeros((u.shape[0], W.shape[0]), dtype=np.uint64)
    for k in range(W.shape[1]):
        tok |= (u[:, W[:, k]].astype(np.uint64) << np.uint64(B * k))
    return _mix64(tok)


def overlap(tok_q, tok_x):
    return (tok_q == tok_x).sum(1)


def operating_point(close_scores, far_scores, T, eps):
    """Smallest t with FMR <= eps; returns (t, FMR, FNMR) or None."""
    for t in range(T + 1):
        fmr = float((far_scores >= t).mean())
        if fmr <= eps:
            return t, fmr, float((close_scores < t).mean())
    return None
