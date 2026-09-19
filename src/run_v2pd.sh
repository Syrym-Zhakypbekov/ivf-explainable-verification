#!/usr/bin/env bash
# Patient-disjoint sensitivity run (v2pd).
# Required restricted inputs: data/2023.xlsx, data/2024.xlsx, data/2025.xlsx.
# Цепочка короче, чем run_v2.sh: 05 → 07 → 09b (без полусинтетики 08, врачей 10 и фигур 20).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export IVF_BASE="${IVF_BASE:-$ROOT}"
export OUT_DIR="${OUT_DIR:-$ROOT/results/reproduced/v2pd}"
export COHORT="${COHORT:-v2pd}"
export THETA_MODE="${THETA_MODE:-oos}"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PKGS='python313.withPackages(ps: with ps; [pandas numpy scipy scikit-learn openpyxl xgboost matplotlib])'
mkdir -p "$OUT_DIR"
echo "START_V2PD $(date '+%F %T') cohort=$COHORT out=$OUT_DIR theta=$THETA_MODE"
run() {
  echo "=== $1 start $(date '+%F %T')"
  nice -n 10 nix-shell -p "$PKGS" --run "python src/$1"
  rc=$?
  echo "=== $1 end $(date '+%F %T') rc=$rc"
  return $rc
}
run 05_models.py && run 07_calibrated.py && run 09b_stress_fixed.py \
  && echo "DONE_V2PD $(date '+%F %T')" | tee -a "$OUT_DIR/run_log.txt" \
  || { echo "FAILED_V2PD $(date '+%F %T')" | tee -a "$OUT_DIR/run_log.txt"; exit 1; }
