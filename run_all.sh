#!/usr/bin/env bash
# Run every experiment in this repository.
#
#   ./run_all.sh            # quick mode: ~5 minutes, smaller samples, checked loosely
#   ./run_all.sh --full     # full mode: reproduces the published numbers (~20 min)
#
# Quick mode writes to experiments/results_quick/ so it never overwrites the
# published results in experiments/results/. Logs go to logs/, figures to figures/.
# The script ends with scripts/check_results.py, which asserts the results against
# the values quoted in the paper.
set -uo pipefail

cd "$(dirname "$0")"

MODE="quick"
if [ "${1:-}" = "--full" ]; then MODE="full"; fi
PY="${PYTHON:-python3}"

if [ "$MODE" = "quick" ]; then
  OUT=experiments/results_quick
  N=400; SEEDS="1 2"; QN=2000; QQ=10; LOO=--loose
else
  OUT=experiments/results
  N=1000; SEEDS="1 2 3 4 5"; QN=10000; QQ=40; LOO=
fi
mkdir -p logs figures "$OUT"

FAILED=0
say() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }
run() {  # run <label> <log> <command...>
  local label="$1" log="$2"; shift 2
  local t0; t0=$(date +%s)
  if "$@" >"logs/$log" 2>&1; then
    printf "   ok   %-44s %3ds\n" "$label" $(( $(date +%s) - t0 ))
  else
    printf "   FAIL %-44s (see logs/%s)\n" "$label" "$log"
    tail -4 "logs/$log" | sed 's/^/        /'
    FAILED=1
  fi
}

say "environment ($MODE mode -> $OUT)"
"$PY" - <<'EOF'
import sys
print(f"   python {sys.version.split()[0]}")
for m in ("numpy", "matplotlib", "h5py"):
    try:
        mod = __import__(m); print(f"   {m} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"   {m} MISSING")
EOF
"$PY" - <<'EOF' || echo "   libsodium missing: E8-E11 will fail"
import ctypes, ctypes.util
p = ctypes.util.find_library("sodium") or "/opt/homebrew/lib/libsodium.dylib"
ctypes.CDLL(p); print("   libsodium ok")
EOF

say "accuracy layer (numpy only)"
run "E1  overlap vs similarity"        e1_multiseed.log \
    "$PY" experiments/sim/run_multiseed.py --n "$N" --seeds $SEEDS --out "$OUT/predicate_multiseed.json"
run "E2  ablations: expansion/T/D"     e2_ablations.log \
    "$PY" experiments/sim/run_ablations.py --n "$N" --seeds $SEEDS --out "$OUT/ablations.json"
run "E3  near-miss resolution"         e3_boundary.log \
    "$PY" experiments/sim/run_boundary.py --n "$N" --seeds $SEEDS --out "$OUT/boundary.json"
run "E4  transfer curve"               e4_transfer.log \
    "$PY" experiments/sim/run_transfer.py --n "$N" --seeds $SEEDS --out "$OUT/transfer.json"
run "E5  accuracy vs selectivity"      e5_selectivity.log \
    "$PY" experiments/sim/run_selectivity.py --n "$N" --seeds $SEEDS --out "$OUT/selectivity.json"
run "E6  two-stage counting"           e6_twostage.log \
    "$PY" experiments/sim/run_twostage.py --n 3000 --q 20 --seeds 1 --out "$OUT/twostage.json"

if [ -f data/sift-128-euclidean.hdf5 ]; then
  run "E7  real SIFT1M accuracy"       e7_realdata.log \
      "$PY" experiments/sim/run_realdata.py --queries 200 --seeds 1 --out "$OUT/real_sift.json"
else
  say "E7  real SIFT1M accuracy -- SKIPPED (run ./scripts/get_data.sh first)"
fi

say "system layer (needs libsodium)"
run "E8  OPRF microbenchmark"          e8_oprf.log \
    "$PY" experiments/proto/bench_oprf.py --out "$OUT/oprf_bench.json"
run "E9  fixed-B prototype"            e9_proto.log \
    "$PY" experiments/proto/qfpsi_proto.py --n "$QN" --queries 30 --out "$OUT/proto_10k.json"
run "E10 two-stage, executed"          e10_twostage_proto.log \
    "$PY" experiments/proto/twostage_proto.py --n "$QN" --q "$QQ" --workers 4 --out "$OUT/twostage_proto.json"
run "E11 Adaptive-B, executed"         e11_adaptive.log \
    "$PY" experiments/proto/adaptive_proto.py --n "$QN" --q 16 --calib 8 --workers 4 --out "$OUT/adaptive_proto.json"

say "figures"
run "Fig 5  overlap vs similarity"     fig5.log \
    "$PY" experiments/sim/make_figure.py "$OUT/predicate_multiseed.json" figures/fig_predicate_v2
run "Fig 6  transfer curve"            fig6.log \
    "$PY" experiments/sim/make_figure_transfer.py "$OUT/transfer.json" figures/fig_transfer_v2

if [ "$MODE" = "full" ]; then
  say "data package (CSV + MANIFEST + SUMMARY)"
  run "regenerate results/csv + SUMMARY" dataset.log \
      "$PY" experiments/make_dataset.py
fi

say "self-check against the numbers quoted in the paper"
"$PY" scripts/check_results.py --results-dir "$OUT" $LOO; CHECK=$?

say "summary"
echo "   mode:     $MODE"
echo "   results:  $OUT"
echo "   logs:     logs/"
echo "   figures:  figures/"
if [ "$FAILED" -ne 0 ] || [ "$CHECK" -ne 0 ]; then
  echo "   STATUS:   FAILED"
  exit 1
fi
echo "   STATUS:   PASS"
