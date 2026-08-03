# -*- coding: utf-8 -*-
"""
Рисунок «Изменение показателя верифицированности при увеличении
интенсивности методологических дефектов» — по ТЗ А.А. Быкова.

Один график, шесть линий (все типы дефектов), общая горизонтальная ось
интенсивности 0–1, горизонтальная линия порога V = θ.

ВАЖНО о шкале интенсивности. Уровни разных дефектов заданы в разных
единицах: ρ (сила связи с утечкой), η (доля подменённого объяснения),
доля обучающей выборки, кратность шума σ, доля повреждённых записей.
Чтобы линии были сопоставимы, интенсивность приведена к общей шкале
0–1 как доля от максимального уровня, испытанного для данного типа
дефекта. Точка 0 у всех линий общая — состояние «дефект отсутствует»,
где V равен значению эталонной модели.

Данные: out/stress2_all.csv (348 конфигураций, независимый тест 2025 г.).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

OUT = Path.home() / "ivf" / "out"

for cand in ("DejaVu Sans", "Liberation Sans", "Noto Sans"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linewidth": 0.7, "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

RU = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
THETA = 0.694


def ru(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


# порядок соответствует перечислению в ТЗ
SPEC = [
    ("T", "Утечка данных",            "#C62828", "o", 3.0),
    ("F", "Подмена объяснений",       "#7B1FA2", "s", 2.4),
    ("S", "Нестабильность",           "#00838F", "^", 2.4),
    ("C", "Предметная инверсия",      "#EF6C00", "D", 2.4),
    ("R", "Неопределённость",         "#546E7A", "v", 2.4),
    ("D", "Повреждение данных",       "#B71C1C", "X", 3.0),
]


def main():
    res = pd.read_csv(OUT / "stress2_all.csv")
    v0 = float(res.loc[res.defective == 0, "V"].mean())   # эталон = интенсивность 0
    print(f"V эталона (интенсивность 0): {ru(v0)}")

    v_best = float(res.loc[res.defective == 0, "V"].max())  # лучший эталон
    fig, ax = plt.subplots(figsize=(9.6, 6.6))

    for t, label, col, mk, lw in SPEC:
        sub = (res[(res["тип"] == t) & (res.defective == 1)]
               .groupby("уровень")["V"].mean().sort_index())
        if not len(sub):
            continue
        # общая шкала: доля от максимального испытанного уровня
        lv = sub.index.values.astype(float)
        x = lv / lv.max()
        # добавляем точку «дефекта нет»
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[v0], sub.values])

        ax.plot(x, y, mk + "-", color=col, lw=lw, ms=8, mec="white", mew=1.4,
                label=label, zorder=4, alpha=0.95)

        # отметить пересечение порога
        below = np.where(y < THETA)[0]
        if len(below) and below[0] > 0:
            i = below[0]
            xc = np.interp(THETA, [y[i], y[i - 1]], [x[i], x[i - 1]])
            ax.plot([xc], [THETA], "o", ms=11, mfc="white", mec=col,
                    mew=2.4, zorder=6)

    # порог
    ax.axhline(THETA, ls="--", color="#212121", lw=2.0, zorder=3)
    ax.text(1.005, THETA, f"  порог верификации\n  V = θ = {ru(THETA)}",
            fontsize=10.5, color="#212121", fontweight="bold", va="center")

    # зоны — подписи справа, где нет данных
    ax.axhspan(THETA, 1.0, color="#2E7D32", alpha=0.055, zorder=0)
    ax.axhspan(0.0, THETA, color="#C62828", alpha=0.045, zorder=0)
    ax.text(0.985, 0.845, "ЗОНА ВЕРИФИКАЦИИ", fontsize=9.5, color="#2E7D32",
            fontweight="bold", alpha=0.9, ha="right")
    ax.text(0.985, 0.035, "ЗОНА ОТКЛОНЕНИЯ", fontsize=9.5, color="#C62828",
            fontweight="bold", alpha=0.9, ha="right")

    # диапазон эталонных моделей: показывает, что порог задан строго
    ax.plot([0], [v_best], "*", ms=15, mfc="#2E7D32", mec="white", mew=1.2, zorder=7)
    ax.plot([0], [v0], "o", ms=12, mfc="white", mec="#212121", mew=2.4, zorder=7)
    ax.annotate(f"дефект отсутствует:\nлучший эталон {ru(v_best)}\nсреднее по эталонам {ru(v0)}",
                (0, v_best), xytext=(20, 34), textcoords="offset points", fontsize=9.5,
                fontweight="bold", color="#212121",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", lw=0.9, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#212121", lw=1.4))

    ax.set_xlabel("Интенсивность методологического дефекта\n"
                  "(0 — дефект отсутствует, 1 — максимальный испытанный уровень)",
                  fontsize=11.5)
    ax.set_ylabel("Показатель верифицированности V", fontsize=11.5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.035, 0.90)
    ax.xaxis.set_major_formatter(RU)
    ax.yaxis.set_major_formatter(RU)
    # легенда под графиком — не перекрывает линии
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              fontsize=10, framealpha=0, title="Тип дефекта", title_fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

    ax.set_title("Изменение показателя верифицированности при увеличении "
                 "интенсивности методологических дефектов",
                 fontsize=13.5, fontweight="bold", loc="left", pad=14)

    fig.text(0.5, -0.30,
             "Утечка данных (T) и повреждение данных (D) обнуляют показатель уже при "
             "минимальной интенсивности — срабатывает вето.\nПодмена объяснений (F) и "
             "предметная инверсия (C) снижают показатель монотонно.",
             ha="center", fontsize=10, style="italic", color="#333")
    fig.text(0.5, -0.395,
             "Уровни разных дефектов заданы в разных единицах (ρ, η, доля выборки, кратность σ), "
             "поэтому интенсивность приведена\nк общей шкале как доля от максимального испытанного "
             "уровня.",
             ha="center", fontsize=9, color="#666")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_intensity.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    print("сохранено: fig7_intensity")


if __name__ == "__main__":
    main()
