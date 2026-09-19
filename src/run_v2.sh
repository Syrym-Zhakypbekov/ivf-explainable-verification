#!/usr/bin/env bash
# Full private-data recomputation for cohort v2.
# Required restricted inputs: data/2023.xlsx, data/2024.xlsx, data/2025.xlsx.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export IVF_BASE="${IVF_BASE:-$ROOT}"
export OUT_DIR="${OUT_DIR:-$ROOT/results/reproduced/v2}"
export COHORT="${COHORT:-v2}"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PKGS='python313.withPackages(ps: with ps; [pandas numpy scipy scikit-learn openpyxl xgboost matplotlib])'
mkdir -p "$OUT_DIR"
echo "START_V2 $(date '+%F %T') cohort=$COHORT out=$OUT_DIR"
run() {
  echo "=== $1 start $(date '+%F %T')"
  nice -n 10 nix-shell -p "$PKGS" --run "python src/$1"
  rc=$?
  echo "=== $1 end $(date '+%F %T') rc=$rc"
  return $rc
}
run 05_models.py && run 07_calibrated.py && run 08_semisynth.py && run 09b_stress_fixed.py \
  && run 10_doctors.py && run 20_figures_en.py && echo "DONE_V2 $(date '+%F %T')" | tee -a "$OUT_DIR/run_log.txt" \
  || { echo "FAILED_V2 $(date '+%F %T')" | tee -a "$OUT_DIR/run_log.txt"; exit 1; }
