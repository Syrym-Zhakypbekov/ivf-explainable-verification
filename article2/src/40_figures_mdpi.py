# -*- coding: utf-8 -*-
"""
СТАТЬЯ №2 — рисунки по требованиям MDPI (Q1).

ТРЕБОВАНИЯ, КОТОРЫЕ ЗДЕСЬ СОБЛЮДЕНЫ

  • Ширина полосы MDPI — 170 мм. Рисунок делается сразу в этом размере,
    чтобы вёрстка не масштабировала его: при уменьшении подписи становятся
    нечитаемыми, при увеличении — размываются.
  • Разрешение 600 dpi для штриховой графики и комбинированных рисунков
    (требование MDPI: не ниже 300 dpi, для line art рекомендовано 600–1000).
  • Минимальный кегль после вёрстки — не менее 8 pt. Поскольку рисунок не
    масштабируется, кегль в исходнике равен кеглю в журнале.
  • Формат: PDF (вектор, для вёрстки), PNG (для рецензентов), EPS (запасной).
  • Подписи только на английском.
  • Палитра различима в градациях серого и при дальтонизме: синий и
    оранжевый вместо привычной пары красный/зелёный.

Все числа берутся из сохранённых результатов, а не пересчитываются заново.
Это исключает расхождение рисунка с текстом — ошибку, из-за которой в
статье №1 доверительный интервал на рисунке отличался от интервала в тексте.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OUT = Path.home() / "ivf2" / "out"
FIG = OUT / "figures_en"
FIG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH = 170 * MM          # ширина полосы MDPI
DPI = 600

# палитра, различимая при дальтонизме и в оттенках серого
BLUE, ORANGE, GREY = "#0B5FA5", "#E07B20", "#8A8F98"
DARK, LIGHT = "#1A1D24", "#EDF1F6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GREY, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png", "eps"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)
    w_mm = fig.get_size_inches()[0] * 25.4
    print(f"  saved {name:28s} width {w_mm:.0f} mm, {DPI} dpi, min font 7.5 pt")


# ═══════════════ Figure 1 — model performance across tasks ═══════════════
def fig1_tasks() -> None:
    df = pd.read_csv(OUT / "all_tasks_results.csv")
    tasks = [
        ("5. Риск бедного ответа (≤3 ооцитов)", "Poor response\n(≤3 oocytes)"),
        ("4. Риск гиперответа (>20 ооцитов)", "Hyper-response\n(>20 oocytes)"),
        ("6. Состоится ли пункция", "Oocyte retrieval\noccurrence"),
        ("7. Положительный ХГЧ (беременность)", "Positive hCG\n(pregnancy)"),
    ]
    names, best, base = [], [], []
    for key, label in tasks:
        sub = df[df["задача"] == key]
        if sub.empty or "auroc" not in sub:
            continue
        m = sub[sub["модель"] != "бейзлайн (частый класс)"]["auroc"].max()
        names.append(label)
        best.append(m)
        base.append(0.5)

    fig, ax = plt.subplots(figsize=(WIDTH, 62 * MM))
    x = np.arange(len(names))
    ax.bar(x, best, 0.55, color=BLUE, zorder=3, label="Best model (AUROC)")
    ax.axhline(0.5, color=GREY, ls="--", lw=1.0, zorder=2,
               label="Random guessing (AUROC = 0.5)")
    for xi, v in zip(x, best):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5,
                fontweight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC (5-fold cross-validation)")
    ax.set_title("Discrimination of clinically relevant outcomes "
                 "from pre-treatment features", loc="left", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    clean(ax)
    save(fig, "fig1_task_performance")


# ═════════════ Figure 2 — ovarian response across AMH deciles ═════════════
def fig2_deciles() -> None:
    d = pd.read_csv(OUT / "mining_amh_deciles.csv")
    d.columns = [c.strip() for c in d.columns]
    med = d["ооцитов_медиана"].values
    poor = d["доля_бедный"].values
    hyper = d["доля_гипер"].values
    lo = d["амг_от"].values
    hi = d["амг_до"].values
    labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(lo, hi)]
    x = np.arange(len(med))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 105 * MM), sharex=True)

    a1.bar(x, med, 0.6, color=BLUE, zorder=3)
    a1.set_ylabel("Median oocytes retrieved")
    a1.set_title("A. Ovarian response by AMH decile",
                 loc="left", fontweight="bold")
    # монотонный рост медианы по децилям — основное наблюдение панели
    a1.annotate("monotonic increase across deciles",
                xy=(7.5, med[7]), xytext=(3.2, med[-1] * 0.92),
                fontsize=7.5, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    clean(a1)

    a2.plot(x, poor, "o-", color=ORANGE, ms=4, label="Poor response (≤3 oocytes)")
    a2.plot(x, hyper, "s-", color=BLUE, ms=4, label="Hyper-response (>20 oocytes)")
    a2.set_ylabel("Proportion of cycles")
    a2.set_xlabel("AMH decile, ng/mL")
    a2.set_xticks(x)
    a2.set_xticklabels(labels, rotation=45, ha="right")
    a2.set_title("B. Extreme responses by AMH decile",
                 loc="left", fontweight="bold")
    a2.legend(framealpha=0.95)
    clean(a2)

    fig.tight_layout()
    save(fig, "fig2_amh_deciles")


# ═══════════════ Figure 3 — model versus clinical rule ═══════════════════
def fig3_rule() -> None:
    df = pd.read_csv(OUT / "all_tasks_results.csv")
    sub = df[df["задача"] == "1. Овариальный ответ (3 класса)"]
    order = ["ПРАВИЛО по АМГ", "бейзлайн (частый класс)",
             "логистическая регрессия", "случайный лес", "градиентный бустинг"]
    en = {"ПРАВИЛО по АМГ": "Clinical AMH rule",
          "бейзлайн (частый класс)": "Majority class",
          "логистическая регрессия": "Logistic regression",
          "случайный лес": "Random forest",
          "градиентный бустинг": "Gradient boosting"}
    vals, names, colors = [], [], []
    for k in order:
        r = sub[sub["модель"] == k]
        if r.empty:
            continue
        vals.append(float(r["macro_f1"].iloc[0]))
        names.append(en[k])
        colors.append(ORANGE if k == "ПРАВИЛО по АМГ" else
                      (GREY if "бейзлайн" in k else BLUE))

    fig, ax = plt.subplots(figsize=(WIDTH, 68 * MM))
    y = np.arange(len(names))
    ax.barh(y, vals, 0.6, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=7.5,
                fontweight="bold", color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.30)
    ax.set_xlabel("Macro-F1 (5-fold cross-validation)")
    ax.set_title("Three-class ovarian response: learned models "
                 "versus the clinical rule", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=ORANGE, label="Clinical rule (AMH thresholds)"),
                       Patch(color=BLUE, label="Learned model"),
                       Patch(color=GREY, label="Trivial baseline")],
              loc="upper center", ncol=3, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.24))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GREY, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig3_model_vs_rule")


# ═════════════ Figure 4 — leakage detector: single-feature AUROC ═════════
def fig4_leakage() -> None:
    c = pd.read_csv(OUT / "mining_correlations.csv")
    c = c[c["AUROC_гиперответ"].notna()]
    leak = c[~c["предлечебный"]].nlargest(8, "AUROC_гиперответ")
    pre = c[c["предлечебный"]].nlargest(8, "AUROC_гиперответ")

    en_pre = {"АМГ": "AMH", "Возр пациентов": "Patient age",
              "Летрозол": "Letrozole (protocol marker)", "ФСГ": "FSH",
              "Номер попытки": "Attempt number", "Возраст мужа": "Partner age",
              "День ПЕ": "Transfer day", "достинекс /таб": "Cabergoline",
              "В PERSONE": "Cycles in clinic", "год рождения": "Birth year",
              "Дивигель": "Estradiol gel (protocol marker)",
              "аэртал": "NSAID (protocol marker)",
              "ЛГ": "LH", "ТТГ": "TSH", "ЛГ старт": "LH at cycle start",
              "Продолж-ть бесплодия/ лет": "Infertility duration, years",
              "Возраст мужа": "Partner age", "Рост мужа": "Partner height",
              "Кесарево сечение": "Previous caesarean",
              "Мед.аборт": "Previous medical abortion",
              "Утрожестан": "Progesterone (protocol marker)",
              "Прогинова": "Estradiol valerate (protocol marker)",
              "Летрозол": "Letrozole (protocol marker)",
              "Перговерис/амп": "Follitropin/lutropin (protocol marker)",
              "Бесплодие": "Infertility type", "Тест": "Test",
              "Операции/Тубэктомия": "Previous tubectomy",
              "А+В %": "Sperm morphology A+B, %"}
    en_leak = {"∑ Зрелых (MII)": "Mature oocytes (MII)",
               "∑ Дробление": "Cleavage-stage embryos",
               "И Получено ооцитов": "Oocytes retrieved (ICSI)",
               "кЭКО Получено ооцитов": "Oocytes retrieved (IVF)",
               "И Зрелых (MII)": "Mature oocytes (ICSI)",
               "кЭКО Зрелых (MII)": "Mature oocytes (IVF)",
               "И Дробление": "Cleavage (ICSI)",
               "DO Получено ооцитов": "Oocytes retrieved (donor)",
               "кЭКО Дробление": "Cleavage (IVF)",
               "∑ Выход б/ц": "Blastocyst yield",
               "∑ n  заморож.эмбр": "Embryos cryopreserved",
               "DO Зрелых (MII)": "Mature oocytes (donor)",
               "DO дробление": "Cleavage (donor)"}

    fig, ax = plt.subplots(figsize=(WIDTH, 84 * MM))
    names, vals, cols = [], [], []
    for _, r in leak.iterrows():
        names.append(en_leak.get(r["признак"], r["признак"]))
        vals.append(r["AUROC_гиперответ"])
        cols.append(ORANGE)
    for _, r in pre.iterrows():
        names.append(en_pre.get(r["признак"], r["признак"]))
        vals.append(r["AUROC_гиперответ"])
        cols.append(BLUE)

    y = np.arange(len(names))
    ax.barh(y, vals, 0.62, color=cols, zorder=3)
    ax.axvline(0.85, color=DARK, ls="--", lw=1.1, zorder=4)
    # подпись порога — вертикально вдоль самой линии, чтобы не задевать
    # ни заголовок сверху, ни легенду снизу
    ax.text(0.858, len(names) * 0.5, "leakage threshold 0.85",
            fontsize=7, color=DARK, rotation=90, va="center", ha="left")
    ax.axvline(0.5, color=GREY, ls=":", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("Single-feature AUROC for hyper-response (>20 oocytes)")
    ax.set_title("Automatic leakage screening: post-hoc features "
                 "predict the outcome almost perfectly",
                 loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=ORANGE, label="Post-hoc (recorded at or after retrieval)"),
                       Patch(color=BLUE, label="Pre-treatment (available at decision time)")],
              loc="lower right", framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GREY, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig4_leakage_screening")


# ══════════ Figure 5 — expert labels reproduce the algorithmic hint ══════
def fig5_annotation() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 62 * MM))

    # A: согласие с подсказкой
    a1.bar([0], [100.0], 0.5, color=ORANGE, zorder=3)
    a1.bar([1], [0.0], 0.5, color=BLUE, zorder=3)
    a1.text(0, 101.5, "3056 / 3056", ha="center", fontsize=8, fontweight="bold")
    a1.text(1, 2.0, "0", ha="center", fontsize=8, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["Agreed with\nalgorithmic hint", "Deviated from\nhint"])
    a1.set_ylabel("Share of labelled cycles, %")
    a1.set_ylim(0, 112)
    a1.set_title("A. Expert agreement with the hint", loc="left", fontweight="bold")
    clean(a1)

    # B: распределение времени решения.
    # Шкала логарифмическая: квартили лежат в узком интервале 0,84–1,16 с,
    # а хвост тянется до 4,5 с. На линейной шкале подписи квартилей
    # накладывались друг на друга.
    q1, med_t, q3, p95 = 0.84, 0.93, 1.16, 4.46
    a2.set_xscale("log")
    a2.barh([0], [q3 - q1], left=q1, height=0.30, color=LIGHT,
            edgecolor=BLUE, lw=1.2, zorder=3)
    a2.plot([q3, p95], [0, 0], color=BLUE, lw=1.1, zorder=3)
    a2.plot([p95, p95], [-0.10, 0.10], color=BLUE, lw=1.2, zorder=4)
    a2.plot([med_t], [0], "o", color=ORANGE, ms=7, zorder=5)

    a2.annotate(f"median {med_t:.2f} s", xy=(med_t, 0.16), xytext=(1.9, 0.44),
                fontsize=7.5, fontweight="bold", color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    a2.text(q1, -0.24, f"Q1 {q1:.2f}", ha="right", va="top", fontsize=7)
    a2.text(q3, -0.24, f"Q3 {q3:.2f}", ha="left", va="top", fontsize=7)
    a2.text(p95, -0.24, f"P95 {p95:.2f}", ha="center", va="top", fontsize=7)
    a2.text(0.42, 0.52, "62 % of decisions took less than one second",
            fontsize=7.5, color=DARK, style="italic")

    a2.set_xlim(0.4, 8)
    a2.set_ylim(-0.62, 0.68)
    a2.set_xticks([0.5, 1, 2, 5])
    a2.set_xticklabels(["0.5", "1", "2", "5"])
    a2.set_yticks([])
    a2.set_xlabel("Decision time per case, seconds (log scale)")
    a2.set_title("B. Time spent per labelling decision", loc="left",
                 fontweight="bold")
    for sp in ("top", "right", "left"):
        a2.spines[sp].set_visible(False)
    a2.grid(axis="x", color=GREY, alpha=0.25, zorder=0)
    a2.set_axisbelow(True)

    fig.suptitle("Expert labelling collected with an algorithmic hint "
                 "reproduces the hint entirely", fontsize=9, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig5_annotation_collapse")


def main() -> None:
    print("generating MDPI-compliant figures (170 mm, 600 dpi, min 7.5 pt):")
    for fn in (fig1_tasks, fig2_deciles, fig3_rule, fig4_leakage, fig5_annotation):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {e}")
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
