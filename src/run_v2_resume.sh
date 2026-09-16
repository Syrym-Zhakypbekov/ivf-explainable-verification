#!/usr/bin/env bash
# продолжение v2 после падения 09b (16.09 16:13): 05/07/08 уже в out_v2
set -u; cd "$HOME/ivf" || exit 1
export OUT_DIR="$HOME/ivf/out_v2" COHORT=v2 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
PKGS='python313.withPackages(ps: with ps; [pandas numpy scipy scikit-learn openpyxl xgboost matplotlib])'
run() { echo "=== $1 start $(date "+%F %T")"; nice -n 10 nix-shell -p "$PKGS" --run "python src/$1"; rc=$?; echo "=== $1 end $(date "+%F %T") rc=$rc"; return $rc; }
echo "RESUME_V2 $(date "+%F %T")"
run 09b_stress_fixed.py && run 10_doctors.py && run 20_figures_en.py && echo "DONE_V2 $(date "+%F %T")" || echo "FAILED_V2 $(date "+%F %T")"
