"""
Minimal Ristretto255 bindings over libsodium, by ctypes.

Only what the QFPSI prototype needs:
  * from_hash        -> hash-to-group for H1
  * scalarmult       -> blinding, server evaluation, unblinding
  * scalar_random / scalar_invert / scalar_mul
  * is_valid_point
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib

_lib_path = ctypes.util.find_library("sodium") or "/opt/homebrew/lib/libsodium.dylib"
_lib = ctypes.CDLL(_lib_path)
_lib.sodium_init()

ELEM = 32   # group element
SCALAR = 32
HASHLEN = 64

for fn in ("crypto_core_ristretto255_from_hash",
           "crypto_scalarmult_ristretto255",
           "crypto_core_ristretto255_scalar_random",
           "crypto_core_ristretto255_scalar_invert",
           "crypto_core_ristretto255_scalar_mul",
           "crypto_core_ristretto255_is_valid_point"):
    getattr(_lib, fn).restype = ctypes.c_int


def h1(token: bytes) -> bytes:
    """H1: arbitrary bytes -> 64-byte hash, then a group element."""
    h = hashlib.blake2b(token, digest_size=HASHLEN).digest()
    out = ctypes.create_string_buffer(ELEM)
    rc = _lib.crypto_core_ristretto255_from_hash(out, h)
    if rc != 0:
        raise ValueError("from_hash failed")
    return out.raw


def scalarmult(n: bytes, p: bytes) -> bytes:
    out = ctypes.create_string_buffer(ELEM)
    rc = _lib.crypto_scalarmult_ristretto255(out, n, p)
    if rc != 0:
        raise ValueError("scalarmult rejected input")
    return out.raw


def scalar_random() -> bytes:
    out = ctypes.create_string_buffer(SCALAR)
    _lib.crypto_core_ristretto255_scalar_random(out)
    return out.raw


def scalar_invert(n: bytes) -> bytes:
    out = ctypes.create_string_buffer(SCALAR)
    _lib.crypto_core_ristretto255_scalar_invert(out, n)
    return out.raw


def scalar_mul(a: bytes, b: bytes) -> bytes:
    out = ctypes.create_string_buffer(SCALAR)
    _lib.crypto_core_ristretto255_scalar_mul(out, a, b)
    return out.raw


def is_valid_point(p: bytes) -> bool:
    return _lib.crypto_core_ristretto255_is_valid_point(p) == 0


def tag(token: bytes, unblinded: bytes) -> bytes:
    """H2: domain-separated 32-byte tag."""
    return hashlib.blake2b(b"QFPSI-H2" + token + unblinded, digest_size=32).digest()
