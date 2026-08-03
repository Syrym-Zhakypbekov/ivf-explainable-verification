# -*- coding: utf-8 -*-
"""
Два рисунка по ТЗ А.А. Быкова.

Рисунок 8. Прогнозная точность и восстановление эталонного объяснения
           на полусинтетических данных. Две панели: A — Macro-F1,
           B — Precision@3 и Recall@3.
           Вывод: точность практически одинакова, качество объяснения
           различается принципиально.

Рисунок 9. Сравнение способности показателей выявлять методологически
           дефектные модели. ROC-кривые для интегрального показателя V,
           Macro-F1, уверенности модели и каждого компонента в отдельности.
           Данные — расширенный эксперимент (348 конфигураций),
           AUROC с доверительными интервалами (кластерный бутстрэп).
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
from sklearn.metrics import roc_auc_score, roc_curve

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


def ru(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("сохранено:", name)


# ─────────────────────── Рисунок 8: полусинтетика ───────────────────────
def fig_semisynth():
    df = pd.read_csv(OUT / "semisynth_recovery.csv")
    names = {"LogReg": "Логистическая\nрегрессия", "RandomForest": "Случайный\nлес",
             "HistGB": "Градиентный\nбустинг"}
    algos = ["LogReg", "RandomForest", "HistGB"]
    d = df.set_index("алгоритм")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.6))
    x = np.arange(len(algos))

    # ── Панель A: точность ──
    f1 = [d.loc[a, "macro_f1_synth"] for a in algos]
    bars = a1.bar(x, f1, 0.52, color="#1565C0", edgecolor="white", lw=1.2)
    a1.bar_label(bars, labels=[ru(v) for v in f1], padding=4,
                 fontsize=11.5, fontweight="bold")
    # ось обрезана у 0,78: иначе разница 0,011 визуально неразличима
    a1.set_ylim(0.78, 0.845)
    a1.set_ylabel("Macro-F1", fontsize=11)
    a1.set_title("A. Прогнозная точность", fontsize=12.5, fontweight="bold", loc="left")
    a1.set_xticks(x); a1.set_xticklabels([names[a] for a in algos], fontsize=10.5)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    a1.axhspan(min(f1), max(f1), color="#C62828", alpha=0.09, zorder=0)
    a1.axhline(min(f1), ls=":", color="#C62828", lw=1.3, zorder=2)
    a1.axhline(max(f1), ls=":", color="#C62828", lw=1.3, zorder=2)
    a1.text(1.0, 0.838, f"весь разброс точности — {ru(max(f1) - min(f1))}",
            fontsize=10.5, color="#C62828", fontweight="bold", ha="center")
    a1.text(0.02, 0.02, "ось обрезана снизу", fontsize=8.5, color="#888",
            style="italic", transform=a1.transAxes)
    clean(a1)

    # ── Панель B: восстановление объяснения ──
    w = 0.30
    p3 = [d.loc[a, "Precision@3"] for a in algos]
    r3 = [d.loc[a, "Recall@3"] for a in algos]
    sa = [d.loc[a, "SignAccuracy"] for a in algos]
    b1 = a2.bar(x - w, p3, w, label="Precision@3", color="#2E7D32", edgecolor="white", lw=1.2)
    b2 = a2.bar(x, r3, w, label="Recall@3", color="#66BB6A", edgecolor="white", lw=1.2)
    b3 = a2.bar(x + w, sa, w, label="SignAccuracy", color="#B0BEC5", edgecolor="white", lw=1.2)
    for b, vals in ((b1, p3), (b2, r3), (b3, sa)):
        a2.bar_label(b, labels=[ru(v, 2) for v in vals], padding=3, fontsize=9.5,
                     fontweight="bold")
    a2.set_ylim(0, 1.16)
    a2.set_ylabel("Значение метрики", fontsize=11)
    a2.set_title("B. Восстановление эталонного объяснения", fontsize=12.5,
                 fontweight="bold", loc="left")
    a2.set_xticks(x); a2.set_xticklabels([names[a] for a in algos], fontsize=10.5)
    a2.yaxis.set_major_formatter(RU)
    a2.legend(fontsize=9.5, loc="upper right", framealpha=0.96, ncol=1)
    a2.text(0, 1.09, "полное восстановление\nзаложенного механизма", fontsize=9.5,
            color="#2E7D32", fontweight="bold", ha="center")
    a2.annotate("в объяснение попадают\nшумовые признаки", xy=(1.0, 0.35),
                xytext=(1.55, 0.62), fontsize=9.5, color="#C62828",
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FDECEA", ec="#C62828", lw=1.1),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5))
    clean(a2)

    fig.suptitle("Прогнозная точность и восстановление эталонного объяснения "
                 "на полусинтетических данных",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.055,
             "При практически одинаковой точности (разброс 0,011) качество объяснения различается "
             "принципиально: логистическая регрессия восстанавливает заложенный механизм полностью, "
             "древесные модели — лишь на треть.",
             ha="center", fontsize=10, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig8_semisynth")


# ─────────────────────── Рисунок 9: ROC по всем показателям ───────────────────────
def fig_roc_all(rng):
    res = pd.read_csv(OUT / "stress2_all.csv")
    y = res["defective"].values

    # «уверенность модели» — наивный аналог R из пилотного эксперимента
    res["confidence"] = 1.0 / res["conf_size"].clip(lower=1.0)

    SERIES = [
        ("V",          "Интегральный показатель V", "#1565C0", 3.4, "-"),
        ("macro_f1",   "Macro-F1",                  "#C62828", 2.6, "-"),
        ("confidence", "Уверенность модели",        "#8D6E63", 2.0, "-."),
        ("F",          "Компонент F (объяснение)",  "#7B1FA2", 1.8, "--"),
        ("S",          "Компонент S (устойчивость)", "#00838F", 1.8, "--"),
        ("C",          "Компонент C (согласованность)", "#EF6C00", 1.8, "--"),
        ("R",          "Компонент R (надёжность)",  "#546E7A", 1.8, "--"),
    ]

    # кластерный бутстрэп доверительных интервалов AUROC
    groups = {k: g.index.values for k, g in res.groupby("config")}
    keys = list(groups)
    boots = {col: [] for col, *_ in SERIES}
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        for col, *_ in SERIES:
            try:
                boots[col].append(roc_auc_score(y[idx], 1 - res[col].values[idx]))
            except Exception:
                pass

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 6.0),
                                 gridspec_kw={"width_ratios": [1.05, 1]})

    rows = []
    for col, label, c, lw, ls in SERIES:
        fpr, tpr, _ = roc_curve(y, 1 - res[col].values)
        auc = roc_auc_score(y, 1 - res[col].values)
        b = np.array(boots[col])
        lo, hi = np.percentile(b, [2.5, 97.5])
        # на панели A выделяем главное сравнение, компоненты — приглушённым фоном
        main = col in ("V", "macro_f1")
        a1.plot(fpr, tpr, ls, color=c, lw=lw if main else 1.4,
                alpha=1.0 if main else 0.42, zorder=5 if main else 3,
                label=f"{label} — {ru(auc)}" if main else None)
        rows.append((label, auc, lo, hi, c))

    a1.plot([0, 1], [0, 1], ":", color="#9E9E9E", lw=1.5,
            label="случайное угадывание — 0,500", zorder=2)
    a1.plot([], [], "-", color="#9E9E9E", lw=1.4, alpha=0.6,
            label="отдельные компоненты (5 кривых)")
    a1.set_xlabel("Доля ложных срабатываний", fontsize=11)
    a1.set_ylabel("Доля обнаруженных дефектов", fontsize=11)
    a1.set_title("A. ROC-кривые обнаружения дефектных моделей",
                 fontsize=12.5, fontweight="bold", loc="left")
    a1.legend(loc="lower right", fontsize=9.5, framealpha=0.96)
    a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.03)
    a1.xaxis.set_major_formatter(RU); a1.yaxis.set_major_formatter(RU)
    clean(a1)

    # ── Панель B: AUROC с доверительными интервалами ──
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    for i, (label, auc, lo, hi, c) in enumerate(rows):
        a2.plot([lo, hi], [i, i], color=c, lw=3.0, solid_capstyle="round", zorder=3)
        a2.plot([auc], [i], "o", ms=10, color=c, mec="white", mew=1.6, zorder=4)
        a2.annotate(f"{ru(auc)}  [{ru(lo)}; {ru(hi)}]", (hi, i), xytext=(10, 0),
                    textcoords="offset points", fontsize=9.5, va="center",
                    color=c, fontweight="bold")
    a2.axvline(0.5, ls=":", color="#9E9E9E", lw=1.6)
    a2.text(0.5, len(rows) - 0.42, "случайное\nугадывание", fontsize=8.5,
            color="#757575", ha="center", va="top")
    # вертикальная отсечка: нижняя граница интервала V выше верхней границы Macro-F1
    iv = next(i for i, r in enumerate(rows) if r[0].startswith("Интегральный"))
    im = next(i for i, r in enumerate(rows) if r[0] == "Macro-F1")
    a2.axvline(rows[iv][2], ls="--", color="#1B5E20", lw=1.3, alpha=0.75, zorder=1)
    a2.text(rows[iv][2], -0.55,
            f"нижняя граница V ({ru(rows[iv][2])}) выше верхней границы Macro-F1 "
            f"({ru(rows[im][3])})",
            fontsize=8.5, color="#1B5E20", fontweight="bold", ha="center")
    a2.set_yticks(ys)
    a2.set_yticklabels([r[0] for r in rows], fontsize=10)
    a2.set_xlabel("AUROC (95 % доверительный интервал)", fontsize=11)
    a2.set_title("B. AUROC с доверительными интервалами",
                 fontsize=12.5, fontweight="bold", loc="left")
    a2.set_xlim(0.05, 1.16)
    a2.set_ylim(-0.7, len(rows) - 0.2)
    a2.xaxis.set_major_formatter(RU)
    clean(a2)

    fig.suptitle("Сравнение способности показателей выявлять методологически "
                 "дефектные модели (348 конфигураций)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.045,
             "Интегральный показатель V превосходит и Macro-F1, и любой отдельно взятый компонент — "
             "различающая способность обеспечивается именно совместной непополнительной агрегацией. "
             "Доверительные интервалы получены кластерным бутстрэпом по конфигурациям.",
             ha="center", fontsize=10, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig9_roc_all")

    print("\nAUROC (по возрастанию):")
    for label, auc, lo, hi, _ in rows:
        print(f"  {label:<34} {ru(auc)}  [{ru(lo)}; {ru(hi)}]")


def main():
    rng = np.random.default_rng(20260802)
    fig_semisynth()
    fig_roc_all(rng)


if __name__ == "__main__":
    main()
