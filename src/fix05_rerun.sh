#!/usr/bin/env bash
# перегон 05 (LogReg) для v2 и base + рисунки v2 (16.09 18:05)
set -u; cd "$HOME/ivf" || exit 1
PKGS='python313.withPackages(ps: with ps; [pandas numpy scipy scikit-learn openpyxl xgboost matplotlib])'
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
r(){ echo "=== $2 $1 start $(date +%T)"; COHORT=$2 OUT_DIR=$HOME/ivf/out_$2 nice -n 10 nix-shell -p "$PKGS" --run "python src/$1"; echo "=== $2 $1 end $(date +%T) rc=$?"; }
r 05_models.py v2 && r 20_figures_en.py v2; r 05_models.py base; echo FIX05_DONE
