# -*- coding: utf-8 -*-
"""
РИСУНОК: «Точность модели и показатель верифицированности при утечке данных».

Две панели (по ТЗ А.А. Быкова):
  Панель A — Macro-F1: для каждого из 4 алгоритмов два столбца
             (корректная модель / модель с утечкой).
  Панель B — то же для показателя V.

Смысл: по точности модель с утечкой выглядит существенно ЛУЧШЕ,
но по критерию верификации отклоняется полностью (V = 0).

Данные берутся из уже посчитанного out/table_models.csv — ничего не пересчитывается.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

OUT = Path.home() / "ivf" / "out"

# кириллица: берём любой доступный шрифт с поддержкой
for cand in ("DejaVu Sans", "Liberation Sans", "Noto Sans"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

ALGOS = ["LogReg", "RandomForest", "HistGB", "XGBoost"]
LABELS = ["Логистическая\nрегрессия", "Случайный\nлес", "Градиентный\nбустинг", "XGBoost"]
C_OK, C_BAD = "#2E7D32", "#C62828"


def main():
    df = pd.read_csv(OUT / "table_models.csv")
    ok = df[df.config == "Корректная (Clean)"].set_index("algo")
    bad = df[df.config == "D1 Утечка"].set_index("algo")

    x = np.arange(len(ALGOS))
    w = 0.36
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ── Панель A: Macro-F1 ──
    f1_ok = [ok.loc[a, "macro_f1"] for a in ALGOS]
    f1_bad = [bad.loc[a, "macro_f1"] for a in ALGOS]
    b1 = axA.bar(x - w / 2, f1_ok, w, label="Корректная модель", color=C_OK)
    b2 = axA.bar(x + w / 2, f1_bad, w, label="Модель с утечкой", color=C_BAD)
    # доверительные интервалы
    axA.errorbar(x - w / 2, f1_ok, fmt="none", capsize=4, ecolor="#333",
                 yerr=[np.array(f1_ok) - ok.loc[ALGOS, "ci_low"].values,
                       ok.loc[ALGOS, "ci_high"].values - np.array(f1_ok)])
    axA.errorbar(x + w / 2, f1_bad, fmt="none", capsize=4, ecolor="#333",
                 yerr=[np.array(f1_bad) - bad.loc[ALGOS, "ci_low"].values,
                       bad.loc[ALGOS, "ci_high"].values - np.array(f1_bad)])
    axA.set_title("A. Точность (Macro-F1)", fontsize=13, fontweight="bold", loc="left")
    axA.set_ylabel("Macro-F1")
    axA.set_ylim(0, 1.0)
    for bars in (b1, b2):
        axA.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    # ── Панель B: показатель верифицированности V ──
    v_ok = [ok.loc[a, "V"] for a in ALGOS]
    v_bad = [bad.loc[a, "V"] for a in ALGOS]
    b3 = axB.bar(x - w / 2, v_ok, w, label="Корректная модель", color=C_OK)
    b4 = axB.bar(x + w / 2, v_bad, w, label="Модель с утечкой", color=C_BAD)
    axB.axhline(0.778, ls="--", lw=1.4, color="#555")
    axB.text(len(ALGOS) - 0.45, 0.795, "порог верификации θ = 0.778",
             fontsize=9, color="#555", ha="right")
    axB.set_title("B. Показатель верифицированности V", fontsize=13,
                  fontweight="bold", loc="left")
    axB.set_ylabel("V")
    axB.set_ylim(0, 1.0)
    axB.bar_label(b3, fmt="%.3f", padding=3, fontsize=9)
    axB.bar_label(b4, labels=["V = 0"] * len(ALGOS), padding=3, fontsize=9,
                  fontweight="bold", color=C_BAD)

    for ax in (axA, axB):
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Точность модели и показатель верифицированности при утечке данных",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Модель с утечкой существенно превосходит корректную по Macro-F1, "
             "но полностью отклоняется по критерию верификации (V = 0) на всех алгоритмах.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_leakage_f1_vs_V.{ext}", dpi=300, bbox_inches="tight")
    print("сохранено: fig_leakage_f1_vs_V.png / .pdf")
    print(pd.DataFrame({"алгоритм": ALGOS, "F1 корректная": f1_ok, "F1 утечка": f1_bad,
                        "V корректная": v_ok, "V утечка": v_bad}).to_string(index=False))


if __name__ == "__main__":
    main()
