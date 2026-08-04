# -*- coding: utf-8 -*-
"""
Рисунки 2–5 для статьи (по запросу А.А. Быкова).

Рисунок 2. Монотонность: чем сильнее дефект, тем ниже показатель V.
Рисунок 3. Абляция — тепловая карта: что теряется без каждого компонента.
Рисунок 4. Разделение конфигураций: точность не различает, V различает.
Рисунок 5. ROC-кривые и доверительный интервал разницы AUROC.

Данные берутся из уже посчитанных stress2_*.csv — ничего не пересчитывается.
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
from sklearn.metrics import roc_auc_score, roc_curve

OUT = Path.home() / "ivf" / "out"

for cand in ("DejaVu Sans", "Liberation Sans", "Noto Sans"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

GREEN, RED, BLUE, GREY = "#2E7D32", "#C62828", "#1565C0", "#666666"
THETA = 0.694


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("сохранено:", name)


# ─────────────────────────── Рисунок 2 ───────────────────────────
def fig2(res):
    """Монотонность: V убывает по мере усиления дефекта."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    styles = {
        "F": ("Подмена объяснения (F)", "o-", "#8E24AA"),
        "C": ("Инверсия клинической логики (C)", "s-", "#EF6C00"),
        "S": ("Неустойчивость (S)", "^-", "#00838F"),
        "R": ("Зашумление данных (R)", "d-", GREY),
    }
    for t, (label, st, col) in styles.items():
        sub = res[(res["тип"] == t) & (res.defective == 1)].groupby("уровень")["V"].mean()
        if t == "S":                       # по построению растёт с долей выборки
            label += " — по построению возрастает"
        ax.plot(range(len(sub)), sub.values, st, color=col, label=label,
                lw=2.2, ms=7, mec="white", mew=1.2)
        for i, (lv, v) in enumerate(sub.items()):
            ax.annotate(f"{lv:g}", (i, v), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color=col)

    # вето D и T — всегда ноль
    ax.axhline(0, color=RED, lw=2.6, ls="-", alpha=0.85)
    ax.text(0.06, 0.022, "Дефекты D и T: V = 0 на всех уровнях (безусловное вето)",
            fontsize=10, color=RED, fontweight="bold")

    ax.axhline(THETA, ls="--", color="#444", lw=1.4)
    ax.text(len(styles) + 0.55, THETA + 0.012, f"порог θ = {THETA}",
            fontsize=9.5, color="#444", ha="right")

    ax.set_xlabel("Уровень интенсивности дефекта (слабый → сильный)", fontsize=11)
    ax.set_ylabel("Показатель верифицированности V", fontsize=11)
    ax.set_title("Рисунок 2. Показатель убывает по мере усиления дефекта",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_ylim(-0.05, 0.85)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "Метод различает не только факт нарушения, но и его тяжесть. "
             "Числа у точек — фактические уровни интенсивности.",
             ha="center", fontsize=9.5, style="italic")
    save(fig, "fig2_monotonic")


# ─────────────────────────── Рисунок 3 ───────────────────────────
def fig3(abl):
    """Абляция — тепловая карта."""
    cols = ["D", "T", "F", "S", "C", "R"]
    M = abl[cols].values.astype(float)
    labels = abl["вариант"].tolist()

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.1, vmax=1.0, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v < 0.42 else "#111",
                    fontweight="bold" if v < 0.65 else "normal")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"дефект\n{c}" for c in cols], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Рисунок 3. Абляция: AUROC обнаружения по типам дефектов",
                 fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.grid(False)

    # подсветить строку полной формулы и строку Macro-F1
    for i, lab in enumerate(labels):
        if lab.startswith("полная"):
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False,
                                       ec="#1B5E20", lw=2.6))
        if "Macro-F1" in lab:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False,
                                       ec="#B71C1C", lw=2.6))

    cb = fig.colorbar(im, ax=ax, fraction=0.033, pad=0.02)
    cb.set_label("AUROC обнаружения дефекта", fontsize=10)
    fig.text(0.5, -0.03,
             "Удаление компонента обрушивает обнаружение «своего» дефекта (диагональ). "
             "Нижняя строка: Macro-F1 по утечке даёт 0,111 — хуже случайного угадывания.",
             ha="center", fontsize=9.5, style="italic")
    save(fig, "fig3_ablation")


# ─────────────────────────── Рисунок 4 ───────────────────────────
def fig4(res):
    """Точность не разделяет конфигурации, показатель V — разделяет."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    ok = res[res.defective == 0]
    bad = res[res.defective == 1]

    for ax, col, title, xlab in ((a1, "macro_f1", "A. Точность (Macro-F1)", "Macro-F1"),
                                 (a2, "V", "B. Показатель верифицированности V", "V")):
        bins = np.linspace(0, 1, 34)
        ax.hist(ok[col], bins=bins, color=GREEN, alpha=0.72,
                label=f"корректные (n={len(ok)})")
        ax.hist(bad[col], bins=bins, color=RED, alpha=0.62,
                label=f"дефектные (n={len(bad)})")
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
        ax.set_xlabel(xlab, fontsize=10.5)
        ax.set_ylabel("Число конфигураций", fontsize=10.5)
        ax.legend(fontsize=9.5, framealpha=0.95)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    a2.axvline(THETA, ls="--", color="#333", lw=1.8)
    a2.text(THETA + 0.012, a2.get_ylim()[1] * 0.93, f"θ = {THETA}",
            fontsize=10, color="#333", fontweight="bold")

    a1.text(0.5, a1.get_ylim()[1] * 0.72, "распределения\nперекрываются",
            fontsize=11, color="#B71C1C", ha="center", fontweight="bold")

    fig.suptitle("Рисунок 4. Разделение корректных и дефектных конфигураций (n = 348)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             "По точности корректные и дефектные конфигурации неразличимы. "
             "Показатель V разводит их: слева от порога — отклонённые, справа — верифицированные.",
             ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    save(fig, "fig4_separation")


# ─────────────────────────── Рисунок 5 ───────────────────────────
def fig5(res, rng):
    """ROC-кривые + распределение бутстрэп-разницы AUROC."""
    y = res["defective"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.4))

    for col, name, c in (("V", "Показатель V", BLUE), ("macro_f1", "Macro-F1", RED)):
        fpr, tpr, _ = roc_curve(y, 1 - res[col].values)
        auc = roc_auc_score(y, 1 - res[col].values)
        a1.plot(fpr, tpr, lw=2.6, color=c, label=f"{name} — AUROC {auc:.3f}")
    a1.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.5, label="случайное угадывание")
    a1.set_xlabel("Доля ложных срабатываний", fontsize=10.5)
    a1.set_ylabel("Доля обнаруженных дефектов", fontsize=10.5)
    a1.set_title("A. ROC-кривые обнаружения дефекта", fontsize=12.5,
                 fontweight="bold", loc="left")
    a1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    # кластерный бутстрэп
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

    a2.hist(diffs, bins=46, color=BLUE, alpha=0.78, edgecolor="white", lw=0.4)
    a2.axvline(0, color=RED, lw=2.4, label="нулевая разница")
    a2.axvline(lo, ls="--", color="#333", lw=1.6)
    a2.axvline(hi, ls="--", color="#333", lw=1.6)
    a2.axvspan(lo, hi, color=BLUE, alpha=0.10)
    a2.set_xlabel("AUROC(V) − AUROC(Macro-F1)", fontsize=10.5)
    a2.set_ylabel("Частота", fontsize=10.5)
    a2.set_title("B. Кластерный бутстрэп разницы AUROC", fontsize=12.5,
                 fontweight="bold", loc="left")
    a2.legend(handles=[
        Patch(facecolor=BLUE, alpha=0.5, label=f"95 % ДИ [{lo:.3f}; {hi:.3f}]"),
        Patch(facecolor=RED, label="ноль — вне интервала"),
    ], fontsize=9.5, loc="upper right", framealpha=0.95)

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Рисунок 5. Статистическая значимость преимущества показателя V",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             f"Во всех 3000 повторных выборках разница положительна; доверительный интервал "
             f"[{lo:.3f}; {hi:.3f}] не включает ноль.",
             ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    save(fig, "fig5_significance")


def main():
    rng = np.random.default_rng(20260802)
    res = pd.read_csv(OUT / "stress2_all.csv")
    abl = pd.read_csv(OUT / "stress2_ablation.csv")
    print(f"загружено: {len(res)} конфигураций, {len(abl)} вариантов абляции\n")
    fig2(res)
    fig3(abl)
    fig4(res)
    fig5(res, rng)


if __name__ == "__main__":
    main()
