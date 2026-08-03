# -*- coding: utf-8 -*-
"""
Рисунок 6. Тепловая карта компонентов верификации — по ТЗ А.А. Быкова.

Строки — конфигурации (M0 эталон и пять изолированных дефектов),
столбцы — компоненты T, F, S, C, R и итоговый показатель V.
Цветовая интенсивность отражает величину; в каждой ячейке — число от 0 до 1.

Смысл: по диагонали видно, какой именно компонент проседает при каждом
типе дефекта (F = 0,100 при ложном объяснении; S = 0,481 при
нестабильности; C = 0,555 при инверсии логики; R = 0,100 при
неопределённости), тогда как остальные компоненты остаются в норме.

Данные: out/final_table.csv (базовый эксперимент, независимый тест 2025 г.).
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
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white"})

RU = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
THETA = 0.778


def ru(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


ROWS = [
    ("M0 Корректная",         "M0 — эталон",                    None),
    ("D-T тихая утечка",      "D-T — утечка данных",            "T"),
    ("D-F ложное объяснение", "D-F — ложное объяснение",        "F"),
    ("D-S нестабильная",      "D-S — нестабильность",           "S"),
    ("D-C инверсия логики",   "D-C — инверсия предметной логики", "C"),
    ("D-R неопределённость",  "D-R — неопределённость",         "R"),
]
COLS = ["T", "F", "S", "C", "R", "V"]
TITLES = ["T\nвременная\nдопустимость", "F\nверность\nобъяснения",
          "S\nустойчивость", "C\nпредметная\nсогласованность",
          "R\nнадёжность", "V\nитоговый\nпоказатель"]


def main():
    df = pd.read_csv(OUT / "final_table.csv").set_index("config")
    M = np.array([[df.loc[key, c] for c in COLS] for key, _, _ in ROWS], dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 5.9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    for i, (_, _, tgt) in enumerate(ROWS):
        for j, c in enumerate(COLS):
            v = M[i, j]
            hit = (c == tgt)                       # «свой» компонент дефекта
            ax.text(j, i, ru(v), ha="center", va="center",
                    fontsize=13 if hit else 11.5,
                    color="white" if v < 0.35 else "#111",
                    fontweight="bold" if (hit or c == "V") else "normal")
            if hit:                                 # обвести провалившийся компонент
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="#B71C1C", lw=3.0, zorder=6))

    # отделить итоговый столбец V от компонентов
    ax.axvline(len(COLS) - 1.5, color="#212121", lw=2.6, zorder=7)

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(TITLES, fontsize=10)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, lab, _ in ROWS], fontsize=11)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # рамка строки эталона: отдельные линии, иначе верхняя обрезается краем осей
    ax.set_ylim(len(ROWS) - 0.35, -0.65)
    ax.set_xlim(-0.65, len(COLS) - 0.35)
    x0, x1, y0, y1 = -0.5, len(COLS) - 0.5, -0.5, 0.5
    for xs, ys in (((x0, x1), (y0, y0)), ((x0, x1), (y1, y1)),
                   ((x0, x0), (y0, y1)), ((x1, x1), (y0, y1))):
        ax.plot(xs, ys, color="#1B5E20", lw=2.8, zorder=8,
                solid_capstyle="projecting", clip_on=False)

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Значение компонента", fontsize=10.5)
    cb.ax.yaxis.set_major_formatter(RU)
    cb.outline.set_visible(False)

    ax.set_title("Рисунок 6. Тепловая карта компонентов верификации",
                 fontsize=14, fontweight="bold", loc="left", pad=16)

    fig.text(0.5, -0.055,
             "Каждый дефект снижает ровно «свой» компонент (обведён красным), "
             "остальные компоненты остаются в норме — дефекты изолированы.",
             ha="center", fontsize=10.5, style="italic", color="#333")
    fig.text(0.5, -0.105,
             f"Итоговый показатель: эталон {ru(M[0, 5])} — верифицирован; все дефектные "
             f"конфигурации отклонены при пороге θ = {ru(THETA)}. "
             "Утечка обнуляет V безусловно за счёт вето по компоненту T.",
             ha="center", fontsize=10, color="#444")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_components_heatmap.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print("сохранено: fig6_components_heatmap")
    print(pd.DataFrame(M, index=[l for _, l, _ in ROWS], columns=COLS).to_string())


if __name__ == "__main__":
    main()
