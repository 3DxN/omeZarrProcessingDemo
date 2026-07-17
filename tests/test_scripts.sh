#!/usr/bin/env bash
# Smoke + correctness tests for all four label-writing scripts.
#
# Copies the store to a throwaway dir, runs each script into it with a fixed
# threshold (deterministic), and verifies the output with verify_label.py.
# The real store is never touched. Exits non-zero if any script fails.
#
#   bash tests/test_scripts.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
THR=300

[ -x "$PY" ] || { echo "venv python not found at $PY (run: python -m venv .venv)"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STORE="$WORK/test.zarr"
cp -r "$ROOT/6001240_labels.zarr" "$STORE"

scripts=(threshold_label.py threshold_label_omezarr.py threshold_label_dask.py threshold_label_ngff.py)
names=(lbl_hand lbl_ozp lbl_dask lbl_ngff)

fail=0
for i in "${!scripts[@]}"; do
  s="${scripts[$i]}"; n="${names[$i]}"
  echo "-- $s -> labels/$n --"
  if "$PY" "$ROOT/$s" "$STORE" --label-name "$n" --threshold "$THR" >/dev/null 2>&1; then
    if "$PY" "$ROOT/tests/verify_label.py" "$STORE" "$n" "$THR"; then
      echo "  PASS"
    else
      echo "  FAIL (verification)"; fail=1
    fi
  else
    echo "  FAIL (script errored)"; fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL TESTS PASSED (${#scripts[@]} scripts)"
else
  echo "SOME TESTS FAILED"; exit 1
fi
