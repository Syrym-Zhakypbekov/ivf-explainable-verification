#!/usr/bin/env bash
# Sensitivity analysis: patient-disjoint temporal split (когорта v2pd, 16.09.2026). Запуск с warp:
#   cd ~/ivf && setsid nohup bash src/run_v2pd.sh > ~/ivf/out_v2pd/run_log.txt 2>&1 < /dev/null &
# Прогресс: tail ~/ivf/out_v2pd/run_log.txt ; pgrep -f "bin/python src/"
# Цепочка короче, чем run_v2.sh: 05 → 07 → 09b (без полусинтетики 08, врачей 10 и фигур 20).
set -u
cd "$HOME/ivf" || exit 1
export OUT_DIR="${OUT_DIR:-$HOME/ivf/out_v2pd}"
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
  && echo "DONE_V2PD $(date '+%F %T')" || echo "FAILED_V2PD $(date '+%F %T')"
