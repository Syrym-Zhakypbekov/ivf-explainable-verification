# -*- coding: utf-8 -*-
"""
Рисунки к углублённому анализу (разделы 1–6), формат MDPI.

Требования те же, что и для основных рисунков: ширина полосы 170 мм,
600 dpi, кегль не ниже 7,5 pt, подписи только английские, палитра
различима при дальтонизме и в оттенках серого.

Все значения читаются из сохранённых таблиц результатов, а не
пересчитываются, — иначе рисунок разойдётся с текстом.
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
WIDTH = 170 * MM
DPI = 600
BLUE, ORANGE, GREY = "#0B5FA5", "#E07B20", "#8A8F98"
GREEN, PURPLE = "#2E7D5B", "#6B4C9A"
DARK, LIGHT = "#1A1D24", "#EDF1F6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def clean(ax, axis: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GREY, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png", "eps"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"  saved {name}")


# ════════ Figure 6 — cumulative pregnancy across attempts ════════
def fig6_survival() -> None:
    st = pd.read_csv(OUT / "surv_km_strata.csv")
    rep = json.loads((OUT / "advanced_report.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(WIDTH, 76 * MM))
    en = {"АМГ ≤ 0,6": "AMH ≤ 0.6 ng/mL",
          "АМГ 0,61–2,0": "AMH 0.61–2.0 ng/mL",
          "АМГ > 2,0": "AMH > 2.0 ng/mL"}
    colors = {"АМГ ≤ 0,6": ORANGE, "АМГ 0,61–2,0": GREY, "АМГ > 2,0": BLUE}

    for name, g in st.groupby("strata"):
        g = g.sort_values("attempt")
        ax.step(g["attempt"], g["cumulative_pregnancy"], where="post",
                color=colors.get(name, DARK), lw=2.0,
                label=en.get(name, name), zorder=3)

    ax.set_xlabel("Treatment attempt")
    ax.set_ylabel("Cumulative probability of pregnancy")
    ax.set_xlim(1, 6)
    ax.set_ylim(0, 0.65)
    ax.set_xticks(range(1, 7))
    ax.set_title("Cumulative pregnancy probability by ovarian reserve "
                 "(Kaplan–Meier)", loc="left", fontweight="bold")
    p = rep.get("logrank_p")
    if p is not None:
        ax.text(0.98, 0.06, f"log-rank p = {p:.1e}", transform=ax.transAxes,
                ha="right", fontsize=7.5, style="italic", color=DARK)
    ax.legend(loc="upper left", framealpha=0.95)
    clean(ax)
    save(fig, "fig6_survival_km")


# ════════ Figure 7 — propensity score matching ════════
def fig7_causal() -> None:
    rep = json.loads((OUT / "advanced_report.json").read_text(encoding="utf-8"))
    c = rep.get("causal")
    if not c:
        return
    pairs = pd.read_csv(OUT / "causal_matched_pairs.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 68 * MM),
                                 gridspec_kw={"width_ratios": [1, 1.15]})

    # A: эффект до и после сопоставления
    vals = [c["наивная_разница"], c["эффект_после_сопоставления"]]
    a1.bar([0, 1], vals, 0.5, color=[ORANGE, BLUE], zorder=3)
    a1.axhline(0, color=DARK, lw=0.9, zorder=4)
    for x, v in zip([0, 1], vals):
        a1.text(x, v + (0.18 if v >= 0 else -0.35), f"{v:+.2f}",
                ha="center", fontsize=8, fontweight="bold", color=DARK)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["Naive\ncomparison", "After propensity\nscore matching"])
    a1.set_ylabel("Difference in oocytes retrieved")
    a1.set_ylim(min(vals) - 1.2, max(vals) + 1.2)
    a1.set_title("A. Apparent protocol effect", loc="left", fontweight="bold")
    clean(a1)

    # B: распределения сопоставленных пар
    bins = np.arange(0, 34, 2)
    a2.hist(pairs["treated"], bins=bins, alpha=0.62, color=BLUE,
            label="GnRH antagonist", zorder=3)
    a2.hist(pairs["control"], bins=bins, alpha=0.62, color=ORANGE,
            label="Short protocol", zorder=3)
    a2.set_xlabel("Oocytes retrieved")
    a2.set_ylabel("Matched patients")
    a2.set_title("B. Matched cohorts", loc="left", fontweight="bold")
    a2.text(0.97, 0.72, f"{c['пар']} matched pairs\np = {c['p']:.3f}",
            transform=a2.transAxes, ha="right", fontsize=7.5,
            style="italic", color=DARK)
    a2.legend(framealpha=0.95)
    clean(a2)

    fig.suptitle("Confounding by indication: the protocol effect disappears "
                 "after matching", fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_causal_matching")


# ════════ Figure 8 — SHAP contributions ════════
def fig8_shap() -> None:
    imp = pd.read_csv(OUT / "shap_importance.csv").head(12)
    en = {"amh": "AMH", "age": "Patient age", "fsh": "FSH", "lh": "LH",
          "tsh": "TSH", "prolactin": "Prolactin", "bmi": "BMI",
          "weight": "Weight", "height": "Height", "age_husb": "Partner age",
          "bmi_husb": "Partner BMI", "attempt": "Attempt number",
          "ab_pct": "Sperm A+B, %", "morph_pct": "Sperm morphology, %",
          "infert_dur": "Infertility duration", "mar_test": "MAR test",
          "marriages": "Number of marriages", "hystero_n": "Hysteroscopies",
          "funding_платные": "Self-funded cycle",
          "funding_прочее": "Other funding"}
    names = [en.get(x, x) for x in imp["признак"]]
    vals = imp["средний_модуль_SHAP"].values

    fig, ax = plt.subplots(figsize=(WIDTH, 76 * MM))
    y = np.arange(len(names))
    ax.barh(y, vals, 0.62, color=[BLUE if v < vals[0] else ORANGE for v in vals],
            zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + vals[0] * 0.015, yi, f"{v:.3f}", va="center",
                fontsize=7, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, vals[0] * 1.16)
    ax.set_xlabel("Mean |SHAP value| (contribution to hyper-response risk)")
    ax.set_title("Feature contributions to individual predictions (SHAP)",
                 loc="left", fontweight="bold")
    clean(ax, axis="x")
    save(fig, "fig8_shap_importance")


# ════════ Figure 9 — calibration ════════
def fig9_calibration() -> None:
    summ = pd.read_csv(OUT / "calibration_summary.csv")
    files = {"без": ("Uncalibrated", GREY),
             "Платт": ("Platt scaling", BLUE),
             "изотоническая": ("Isotonic regression", ORANGE)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 72 * MM),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    a1.plot([0, 1], [0, 1], ls=":", color=DARK, lw=1.1,
            label="Perfect calibration", zorder=2)
    for key, (label, col) in files.items():
        f = OUT / f"calib_{key}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        a1.plot(d["средняя_вероятность"], d["фактическая_частота"],
                "o-", color=col, ms=4, label=label, zorder=3)
    a1.set_xlabel("Predicted probability")
    a1.set_ylabel("Observed frequency")
    a1.set_title("A. Reliability diagram", loc="left", fontweight="bold")
    a1.legend(loc="upper left", framealpha=0.95)
    clean(a1)

    en_m = {"без калибровки": "Uncalibrated", "Платт (сигмоида)": "Platt",
            "изотоническая": "Isotonic"}
    names = [en_m.get(m, m) for m in summ["метод"]]
    x = np.arange(len(names))
    w = 0.38
    a2.bar(x - w / 2, summ["Брайер"], w, color=BLUE, label="Brier score", zorder=3)
    a2.bar(x + w / 2, summ["ECE"], w, color=ORANGE,
           label="Expected calibration error", zorder=3)
    for xi, (b, e) in enumerate(zip(summ["Брайер"], summ["ECE"])):
        a2.text(xi - w / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=6.8)
        a2.text(xi + w / 2, e + 0.003, f"{e:.3f}", ha="center", fontsize=6.8)
    a2.set_xticks(x)
    a2.set_xticklabels(names, rotation=12, ha="right")
    a2.set_ylabel("Lower is better")
    a2.set_title("B. Calibration error", loc="left", fontweight="bold")
    a2.legend(framealpha=0.95)
    clean(a2)

    fig.suptitle("Probability calibration for hyper-response risk",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig9_calibration")


# ════════ Figure 10 — phenotype clusters ════════
def fig10_clusters() -> None:
    emb = pd.read_csv(OUT / "cluster_embedding.csv")
    prof = pd.read_csv(OUT / "cluster_profile.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 74 * MM),
                                 gridspec_kw={"width_ratios": [1.1, 1]})

    noise = emb[emb["cluster"] == -1]
    a1.scatter(noise["umap1"], noise["umap2"], s=3, c=GREY, alpha=0.35,
               linewidths=0, zorder=2, label="Unassigned")
    real = emb[emb["cluster"] >= 0]
    sc = a1.scatter(real["umap1"], real["umap2"], s=5, c=real["amh"],
                    cmap="viridis", vmin=0, vmax=6, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=a1, fraction=0.045, pad=0.02)
    cb.set_label("AMH, ng/mL", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    # выбросы растягивают оси и сжимают облако в угол — ограничиваем
    # диапазон перцентилями
    a1.set_xlim(np.percentile(emb["umap1"], 0.5), np.percentile(emb["umap1"], 99.5))
    a1.set_ylim(np.percentile(emb["umap2"], 0.5), np.percentile(emb["umap2"], 99.5))
    a1.set_xlabel("UMAP dimension 1")
    a1.set_ylabel("UMAP dimension 2")
    a1.set_title("A. Patient embedding", loc="left", fontweight="bold")
    a1.legend(loc="lower left", framealpha=0.95, markerscale=2.5)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    p = prof[prof["кластер"] >= 0].sort_values("ооцитов_медиана")
    y = np.arange(len(p))
    a2.barh(y, p["ооцитов_медиана"], 0.6, color=BLUE, zorder=3)
    for yi, (v, n) in enumerate(zip(p["ооцитов_медиана"], p["n"])):
        a2.text(v + 0.25, yi, f"n = {int(n)}", va="center", fontsize=6.8,
                color=DARK)
    a2.set_yticks(y)
    a2.set_yticklabels([f"Cluster {int(c)}" for c in p["кластер"]])
    a2.set_xlabel("Median oocytes retrieved")
    a2.set_xlim(0, p["ооцитов_медиана"].max() * 1.28)
    a2.set_title("B. Response by cluster", loc="left", fontweight="bold")
    clean(a2, axis="x")

    fig.suptitle("Unsupervised patient phenotypes (UMAP + HDBSCAN)",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig10_clusters")


def main() -> None:
    print("generating advanced-analysis figures (170 mm, 600 dpi):")
    for fn in (fig6_survival, fig7_causal, fig8_shap, fig9_calibration,
               fig10_clusters):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
