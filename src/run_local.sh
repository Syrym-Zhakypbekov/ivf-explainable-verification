#!/usr/bin/env bash
# Local private-data recomputation without nix-shell.
# Example: PY=python COHORT=v2 THREADS=8 bash src/run_local.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-python}"
export IVF_BASE="${IVF_BASE:-$ROOT}"
export OUT_DIR="${OUT_DIR:-$ROOT/results/reproduced/local}" COHORT="${COHORT:-v2}" PYTHONIOENCODING=utf-8
T="${THREADS:-8}"; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
mkdir -p "$OUT_DIR"
echo "START_LOCAL $(date '+%F %T') cohort=$COHORT out=$OUT_DIR threads=$T"
run() { echo "=== $1 start $(date '+%F %T')"; "$PY" "src/$1"; rc=$?; echo "=== $1 end $(date '+%F %T') rc=$rc"; return $rc; }
run 05_models.py && run 07_calibrated.py && run 08_semisynth.py && run 09b_stress_fixed.py \
  && run 10_doctors.py && run 20_figures_en.py && echo "DONE_LOCAL $(date '+%F %T')" || echo "FAILED_LOCAL $(date '+%F %T')"
