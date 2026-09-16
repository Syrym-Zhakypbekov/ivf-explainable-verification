#!/usr/bin/env bash
# Пересчёт конвейера на когорте v2 (16.09.2026). Запуск с warp:
#   cd ~/ivf && setsid nohup bash src/run_v2.sh > ~/ivf/out_v2/run_log.txt 2>&1 < /dev/null &
# Прогресс: tail ~/ivf/out_v2/run_log.txt ; pgrep -f "bin/python src/"
set -u
cd "$HOME/ivf" || exit 1
export OUT_DIR="${OUT_DIR:-$HOME/ivf/out_v2}"
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
  && run 10_doctors.py && run 20_figures_en.py && echo "DONE_V2 $(date '+%F %T')" >> "$OUT_DIR/run_log.txt" \
  || echo "FAILED_V2 $(date '+%F %T')"
