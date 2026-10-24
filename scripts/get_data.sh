#!/usr/bin/env bash
# Fetch the real-vector dataset used by experiment E7 (SIFT1M, ~525 MB).
#
# Sources tried on 2026-10-24 from a network in China:
#   ann-benchmarks.com                ~1.5 MB/s   <-- used here
#   ftp://ftp.irisa.fr/local/texmex    9 KB/s    (works, too slow)
#   http://corpus-texmex.irisa.fr      403
#   HuggingFace CDN                   DNS blocked
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL="https://ann-benchmarks.com/sift-128-euclidean.hdf5"
OUT="data/sift-128-euclidean.hdf5"

echo "downloading $URL -> $OUT (resumable, Ctrl-C is safe)"
curl -L -C - --retry 5 --retry-delay 3 -o "$OUT" "$URL"

echo
ls -la "$OUT"
python3 - <<'PY'
import hashlib, h5py, os
p = "data/sift-128-euclidean.hdf5"
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(f"size   {os.path.getsize(p)/2**20:.0f} MiB")
print(f"sha256 {h.hexdigest()}")
with h5py.File(p, "r") as f:
    for k in f.keys():
        print(f"  {k:10s} {f[k].shape}")
PY
echo
echo "now run:  ./run_all.sh          (E7 will be included automatically)"
