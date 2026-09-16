#!/usr/bin/env bash
# Локальный пересчёт на ноуте владельца (i9-13 32 потока, RTX 4090): без nix-shell, интерпретатор из PY.
#   COHORT=base OUT_DIR=~/ivf/out_base THREADS=8 bash src/run_local.sh          # сравнение «было → стало»
#   COHORT=v2   OUT_DIR=~/ivf/out_v2_local THREADS=8 bash src/run_local.sh
# Подготовка (один раз): ~/ivf/src → junction на клон репо; ~/ivf/data/{2023,2024,2025}.xlsx → hardlink на 0192+bykov+wishes (копий PII не делать).
set -u
cd "$HOME/ivf" || exit 1
PY="${PY:-C:/Users/syrym/miniforge3/envs/admission/python.exe}"
export OUT_DIR="${OUT_DIR:-$HOME/ivf/out_local}" COHORT="${COHORT:-v2}" PYTHONIOENCODING=utf-8
T="${THREADS:-8}"; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
mkdir -p "$OUT_DIR"
echo "START_LOCAL $(date '+%F %T') cohort=$COHORT out=$OUT_DIR threads=$T"
run() { echo "=== $1 start $(date '+%F %T')"; "$PY" "src/$1"; rc=$?; echo "=== $1 end $(date '+%F %T') rc=$rc"; return $rc; }
run 05_models.py && run 07_calibrated.py && run 08_semisynth.py && run 09b_stress_fixed.py \
  && run 10_doctors.py && run 20_figures_en.py && echo "DONE_LOCAL $(date '+%F %T')" || echo "FAILED_LOCAL $(date '+%F %T')"
