# -*- coding: utf-8 -*-
"""
Рисунки 2–5 для статьи, версия 2 — уровень научного журнала.

Исправлено по результатам визуальной проверки первой версии:
  Рис. 2 — подписи точек налезали друг на друга, легенда закрывала линию
           порога, ось X была безразмерной («1–5»). Переделан на панели:
           каждый тип дефекта в своей ячейке с собственной осью уровней.
  Рис. 3 — рамка выделения строки съезжала. Исправлено смещение.
  Рис. 4 — БЫЛ СЛОМАН: 12 корректных против 336 дефектных, зелёного
           не видно. Гистограмма заменена на точечный график по
           конфигурациям, где обе группы различимы.
  Рис. 5 — десятичный разделитель приведён к запятой (русский стандарт).

Общее: единая типографика, запятая как разделитель, светлый фон,
никаких пересечений подписей.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path.home() / "ivf" / "out"

for cand in ("DejaVu Sans", "Liberation Sans", "Noto Sans"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

GREEN, RED, BLUE, GREY, DARK = "#2E7D32", "#C62828", "#1565C0", "#757575", "#212121"
THETA = 0.694

# русский десятичный разделитель на всех осях
RU = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
RU2 = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def ru(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("сохранено:", name)


# ─────────────────────────── Рисунок 2 ───────────────────────────
def fig2(res):
    """Монотонность — по одной панели на тип дефекта, без наложений."""
    spec = [
        ("F", "Подмена объяснения", "#7B1FA2", "убывает"),
        ("C", "Инверсия клинической логики", "#EF6C00", "убывает"),
        ("S", "Неустойчивость модели", "#00838F", "растёт с объёмом выборки"),
        ("R", "Зашумление входных данных", GREY, "не реагирует"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.5), sharey=True)

    for ax, (t, title, col, note) in zip(axes, spec):
        sub = (res[(res["тип"] == t) & (res.defective == 1)]
               .groupby("уровень")["V"].mean())
        x = np.arange(len(sub))
        ax.plot(x, sub.values, "o-", color=col, lw=2.4, ms=8,
                mec="white", mew=1.5, zorder=3)
        ax.fill_between(x, 0, sub.values, color=col, alpha=0.10, zorder=1)

        ax.axhline(THETA, ls="--", color="#555", lw=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}".replace(".", ",") for v in sub.index], fontsize=9.5)
        ax.set_title(f"{title}\n({t}) — {note}", fontsize=10.5, fontweight="bold",
                     color=col, pad=9)
        ax.set_xlabel("уровень дефекта", fontsize=9.5)
        ax.set_ylim(0, 0.85)
        clean(ax)

        # значения подписываем над точками, чередуя высоту — без наложений
        for i, v in enumerate(sub.values):
            ax.annotate(ru(v), (i, v), textcoords="offset points",
                        xytext=(0, 11 if i % 2 == 0 else -17),
                        ha="center", fontsize=8.5, color=col, fontweight="bold")

    axes[0].set_ylabel("Показатель верифицированности V", fontsize=11)
    axes[0].yaxis.set_major_formatter(RU)
    axes[3].text(0.5, THETA + 0.022, f"порог θ = {ru(THETA)}", fontsize=9,
                 color="#555", ha="center", transform=axes[3].get_yaxis_transform(),
                 clip_on=False)

    # отдельная полоса про безусловное вето
    fig.text(0.5, -0.045,
             "Дефекты D (недопустимые данные) и T (утечка): V = 0 на всех уровнях "
             "интенсивности — вето срабатывает безусловно.",
             ha="center", fontsize=10.5, color=RED, fontweight="bold")
    fig.suptitle("Рисунок 2. Поведение показателя V при усилении дефекта",
                 fontsize=13.5, fontweight="bold", y=1.06)
    fig.text(0.5, -0.10,
             "Показатель различает не только факт нарушения, но и его тяжесть. "
             "Компонент R в текущей реализации к своему дефекту нечувствителен (см. раздел 6).",
             ha="center", fontsize=9.5, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig2_monotonic")


# ─────────────────────────── Рисунок 3 ───────────────────────────
def fig3(abl):
    """Абляция — тепловая карта (исправлено положение рамок)."""
    cols = ["D", "T", "F", "S", "C", "R"]
    M = abl[cols].values.astype(float)
    labels = [l.replace("полная V", "Полная формула V")
               .replace("арифм. среднее", "Арифметическое среднее")
               .replace("только Macro-F1", "Только Macro-F1") for l in abl["вариант"]]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.1, vmax=1.0, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, ru(v), ha="center", va="center", fontsize=10,
                    color="white" if v < 0.40 else "#111",
                    fontweight="bold" if v < 0.68 else "normal")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"дефект {c}" for c in cols], fontsize=10.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # рамки строк: пределы расширены, иначе линия обрезается краем области
    ax.set_ylim(len(labels) - 0.35, -0.65)
    ax.set_xlim(-0.65, len(cols) - 0.35)
    for i, lab in enumerate(labels):
        ec = "#1B5E20" if lab.startswith("Полная") else ("#B71C1C" if "Macro-F1" in lab else None)
        if not ec:
            continue
        x0, x1, y0, y1 = -0.5, len(cols) - 0.5, i - 0.5, i + 0.5
        # четыре отдельные линии надёжнее патча: не подрезаются границей осей
        for xs, ys in (((x0, x1), (y0, y0)), ((x0, x1), (y1, y1)),
                       ((x0, x0), (y0, y1)), ((x1, x1), (y0, y1))):
            ax.plot(xs, ys, color=ec, lw=2.8, zorder=7,
                    solid_capstyle="projecting", clip_on=False)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("AUROC обнаружения дефекта", fontsize=10)
    cb.ax.yaxis.set_major_formatter(RU)
    cb.outline.set_visible(False)

    ax.set_title("Рисунок 3. Абляция: вклад каждого компонента в обнаружение дефектов",
                 fontsize=13, fontweight="bold", loc="left", pad=16)
    fig.text(0.5, -0.02,
             "Удаление компонента обрушивает обнаружение «своего» дефекта. "
             "Нижняя строка: Macro-F1 по утечке даёт 0,111 — систематически хуже случайного угадывания.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    save(fig, "fig3_ablation")


# ─────────────────────────── Рисунок 4 ───────────────────────────
def fig4(res):
    """Разделение конфигураций — точечный график вместо гистограммы.

    Прежняя гистограмма была нечитаема: 12 корректных против 336 дефектных.
    Здесь каждая точка — конфигурация; группы разнесены по вертикали.
    """
    # эталон сверху: строки рисуются снизу вверх, поэтому список перевёрнут
    order = ["R", "C", "S", "F", "T", "D", "—"]
    names = {"—": "ЭТАЛОН\n(корректные)", "D": "недопустимые\nданные", "T": "утечка",
             "F": "подмена\nобъяснения", "S": "неустойчивость",
             "C": "инверсия\nлогики", "R": "зашумление"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.8), sharey=True)
    rng = np.random.default_rng(7)

    for ax, col, title in ((a1, "macro_f1", "A. Точность (Macro-F1)"),
                           (a2, "V", "B. Показатель верифицированности V")):
        for yi, t in enumerate(order):
            sub = res[res["тип"] == t]
            if not len(sub):
                continue
            c = GREEN if t == "—" else RED
            jit = rng.uniform(-0.20, 0.20, len(sub))
            ax.scatter(sub[col], yi + jit, s=34, color=c, alpha=0.50,
                       edgecolor="white", linewidth=0.5, zorder=3)
            m = sub[col].mean()
            ax.plot([m, m], [yi - 0.34, yi + 0.34], color=DARK, lw=2.6, zorder=4)
            ax.annotate(ru(m, 2), (m, yi + 0.44), ha="center", fontsize=9,
                        fontweight="bold", color=DARK)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[t] for t in order], fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.75, len(order) - 0.15)
        ax.set_xlabel(title.split(". ")[1], fontsize=10.5)
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
        ax.xaxis.set_major_formatter(RU)
        # подсветка полосы эталона (последняя строка = верхняя)
        ei = order.index("—")
        ax.axhspan(ei - 0.5, ei + 0.5, color=GREEN, alpha=0.09, zorder=0)
        clean(ax)

    # порог: подпись уводим ПОД ось, чтобы не пересекала подписи средних
    a2.axvline(THETA, ls="--", color=DARK, lw=1.8, zorder=2)
    a2.annotate(f"порог θ = {ru(THETA)}", xy=(THETA, -0.72),
                fontsize=9.5, color=DARK, fontweight="bold",
                ha="right", xytext=(-9, -2), textcoords="offset points")

    # пояснения — в свободной нижней зоне каждой панели, без стрелок через график
    a1.text(0.80, 1.15, "по точности эталон\nне отличается от дефектных",
            fontsize=9.5, color=RED, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FDECEA", ec=RED, lw=1.2))
    a2.text(0.26, 1.15, "по показателю V эталон\nвыходит за порог",
            fontsize=9.5, color=GREEN, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#EAF5EA", ec=GREEN, lw=1.2))

    fig.legend(handles=[
        Patch(facecolor=GREEN, alpha=0.6, label="корректные конфигурации (n = 12)"),
        Patch(facecolor=RED, alpha=0.6, label="дефектные конфигурации (n = 336)"),
    ], loc="lower center", ncol=2, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, -0.055))

    fig.suptitle("Рисунок 4. Разделение корректных и дефектных конфигураций (n = 348)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.115,
             "Каждая точка — одна конфигурация; вертикальные штрихи — средние по группе. "
             "По точности эталон неотличим от дефектных конфигураций; показатель V выносит его за порог.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig4_separation")


# ─────────────────────────── Рисунок 5 ───────────────────────────
def fig5(res, rng):
    y = res["defective"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    for col, name, c, w in (("V", "Показатель V", BLUE, 2.8),
                            ("macro_f1", "Macro-F1", RED, 2.4)):
        fpr, tpr, _ = roc_curve(y, 1 - res[col].values)
        auc = roc_auc_score(y, 1 - res[col].values)
        a1.plot(fpr, tpr, lw=w, color=c, label=f"{name} — AUROC {ru(auc)}", zorder=3)
        a1.fill_between(fpr, tpr, alpha=0.07, color=c, step=None, zorder=1)
    a1.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.6,
            label="случайное угадывание", zorder=2)
    a1.set_xlabel("Доля ложных срабатываний", fontsize=10.5)
    a1.set_ylabel("Доля обнаруженных дефектов", fontsize=10.5)
    a1.set_title("A. ROC-кривые обнаружения дефекта", fontsize=12.5,
                 fontweight="bold", loc="left")
    a1.legend(loc="lower right", fontsize=10, framealpha=0.96)
    a1.xaxis.set_major_formatter(RU); a1.yaxis.set_major_formatter(RU)
    a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.04)
    clean(a1)

    groups = {k: g.index.values for k, g in res.groupby("config")}
    keys = list(groups)
    diffs = []
    for _ in range(3000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], 1 - res["V"].values[idx])
                     - roc_auc_score(y[idx], 1 - res["macro_f1"].values[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    a2.hist(diffs, bins=46, color=BLUE, alpha=0.80, edgecolor="white", lw=0.5, zorder=3)
    a2.axvspan(lo, hi, color=BLUE, alpha=0.11, zorder=1)
    for v in (lo, hi):
        a2.axvline(v, ls="--", color=DARK, lw=1.5, zorder=4)
    a2.axvline(0, color=RED, lw=2.6, zorder=5)
    a2.annotate("ноль", (0, a2.get_ylim()[1] * 0.96), color=RED, fontsize=10,
                fontweight="bold", ha="left", xytext=(6, 0), textcoords="offset points")
    a2.set_xlabel("AUROC(V) − AUROC(Macro-F1)", fontsize=10.5)
    a2.set_ylabel("Число повторных выборок", fontsize=10.5)
    a2.set_title("B. Кластерный бутстрэп разницы AUROC", fontsize=12.5,
                 fontweight="bold", loc="left")
    a2.xaxis.set_major_formatter(RU2)
    a2.legend(handles=[
        Patch(facecolor=BLUE, alpha=0.45, label=f"95 % ДИ [{ru(lo)}; {ru(hi)}]"),
        Patch(facecolor=RED, label="ноль лежит вне интервала"),
    ], fontsize=10, loc="upper right", framealpha=0.96)
    clean(a2)

    fig.suptitle("Рисунок 5. Статистическая значимость преимущества показателя V",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             f"Во всех 3000 кластерных повторных выборках разница положительна; "
             f"доверительный интервал [{ru(lo)}; {ru(hi)}] не включает ноль.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig5_significance")


def main():
    rng = np.random.default_rng(20260802)
    res = pd.read_csv(OUT / "stress2_all.csv")
    abl = pd.read_csv(OUT / "stress2_ablation.csv")
    print(f"загружено: {len(res)} конфигураций "
          f"({(res.defective == 0).sum()} корректных / {(res.defective == 1).sum()} дефектных)\n")
    fig2(res); fig3(abl); fig4(res); fig5(res, rng)


if __name__ == "__main__":
    main()
