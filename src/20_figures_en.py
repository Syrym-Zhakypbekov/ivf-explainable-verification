# -*- coding: utf-8 -*-
"""
All figures in ENGLISH for journal submission (Scopus).

Fig. 1  Accuracy vs verifiability under data leakage (4 algorithms)
Fig. 2  Heat map of verification components by configuration
Fig. 3  Statistical significance: ROC curves + bootstrap CI of AUROC difference
Fig. 4  Separation of correct and defective configurations (n = 348)
Fig. 5  Verifiability index vs defect intensity (6 defect types, threshold line)
Fig. 6  Component behaviour by defect type (native intensity units)
Fig. 7  Ablation: contribution of each component
Fig. 8  Predictive accuracy vs explanation recovery (semi-synthetic data)
Fig. 9  Detection ability of all indicators, AUROC with confidence intervals

Decimal separator: point (English convention).
Data are read from the already computed CSV files — nothing is recomputed.
"""
from __future__ import annotations

import json
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
EN = OUT / "en"
EN.mkdir(exist_ok=True)

for cand in ("DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linewidth": 0.7, "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    # ── требования журналов (MDPI / Elsevier) ────────────────────────────
    # Шрифты встраиваются как TrueType, а не Type-3: Type-3 запрещён
    # большинством типографий и вызывает подстановку при препрессе.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# MDPI: одноколоночный рисунок ≈ 85 мм, двухколоночный ≈ 170 мм.
MM = 1 / 25.4
W1, W2 = 85 * MM, 170 * MM          # дюймы
DPI_RASTER = 600                     # PNG-превью; основной формат — вектор PDF

GREEN, RED, BLUE, GREY, DARK = "#2E7D32", "#C62828", "#1565C0", "#757575", "#212121"
THETA2, THETA1 = 0.694, 0.778     # thresholds: stress test / pilot experiment


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name, target_mm=170):
    """Сохраняет рисунок в формате, пригодном для подачи в журнал.

    PDF/EPS — вектор (для вектора требование «1000 dpi» неприменимо, качество
    не теряется при любом масштабе). PNG 600 dpi — растровое превью.
    Дополнительно проверяется, что после масштабирования до ширины полосы
    ни одна подпись не окажется мельче 6 pt (MDPI требует различимости,
    практический ориентир — не ниже 8 pt для основного текста рисунка).
    """
    w_in = fig.get_size_inches()[0]
    scale = (target_mm * MM) / w_in          # во сколько раз уменьшится при вёрстке

    sizes = [t.get_fontsize() for ax in fig.get_axes()
             for t in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                       + ax.get_xticklabels() + ax.get_yticklabels() + ax.texts)
             if t.get_text()]
    if sizes:
        eff = min(sizes) * scale
        flag = "" if eff >= 6.0 else f"  ⚠ минимальный кегль {eff:.1f} pt < 6 pt"
    else:
        eff, flag = float("nan"), ""

    fig.savefig(EN / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(EN / f"{name}.eps", bbox_inches="tight", facecolor="white")
    fig.savefig(EN / f"{name}.png", dpi=DPI_RASTER, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {name:<28} ширина {w_in*25.4:.0f} мм → {target_mm} мм, "
          f"мин. кегль после вёрстки {eff:.1f} pt{flag}")


# ═══════════════════════ Fig. 1 — leakage ═══════════════════════
def fig1():
    df = pd.read_csv(OUT / "table_models.csv")
    ok = df[df.config == "Корректная (Clean)"].set_index("algo")
    bad = df[df.config == "D1 Утечка"].set_index("algo")
    algos = ["LogReg", "RandomForest", "HistGB", "XGBoost"]
    labels = ["Logistic\nregression", "Random\nforest", "Gradient\nboosting", "XGBoost"]

    x, w = np.arange(4), 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 3.30))

    f1o = [ok.loc[a, "macro_f1"] for a in algos]
    f1b = [bad.loc[a, "macro_f1"] for a in algos]
    b1 = a1.bar(x - w/2, f1o, w, label="Correct model", color=GREEN, edgecolor="white", lw=0.75)
    b2 = a1.bar(x + w/2, f1b, w, label="Model with leakage", color=RED, edgecolor="white", lw=0.75)
    a1.errorbar(x - w/2, f1o, fmt="none", capsize=2.4, ecolor="#333",
                yerr=[np.array(f1o) - ok.loc[algos, "ci_low"].values,
                      ok.loc[algos, "ci_high"].values - np.array(f1o)])
    a1.errorbar(x + w/2, f1b, fmt="none", capsize=2.4, ecolor="#333",
                yerr=[np.array(f1b) - bad.loc[algos, "ci_low"].values,
                      bad.loc[algos, "ci_high"].values - np.array(f1b)])
    for xs, vals, hi in ((x - w/2, f1o, ok.loc[algos, "ci_high"].values),
                         (x + w/2, f1b, bad.loc[algos, "ci_high"].values)):
        for xi, v, h in zip(xs, vals, hi):
            a1.text(xi, h + 0.028, f"{v:.3f}", ha="center", fontsize=7.0)
    a1.set_title("A. Predictive accuracy (macro-F1)", fontsize=7.8, fontweight="bold", loc="left")
    a1.set_ylabel("Macro-F1", fontsize=7.0)
    a1.set_ylim(0, 1.12)

    vo = [ok.loc[a, "V"] for a in algos]
    vb = [bad.loc[a, "V"] for a in algos]
    b3 = a2.bar(x - w/2, vo, w, label="Correct model", color=GREEN, edgecolor="white", lw=0.75)
    a2.bar(x + w/2, vb, w, label="Model with leakage", color=RED, edgecolor="white", lw=0.75)
    a2.axhline(THETA1, ls="--", lw=1.02, color="#444",
               label=f"verification threshold θ = {THETA1}")
    a2.bar_label(b3, fmt="%.3f", padding=3, fontsize=7.0)
    for xi in x:
        a2.text(xi + w/2, 0.03, "V = 0", ha="center", fontsize=7.0,
                fontweight="bold", color=RED)
    a2.set_title("B. Verifiability index V", fontsize=7.8, fontweight="bold", loc="left")
    a2.set_ylabel("V", fontsize=7.0)
    a2.set_ylim(0, 1.12)

    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0)
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.96)
        clean(ax)

    fig.suptitle("Predictive accuracy and verifiability index under data leakage",
                 fontsize=8.4, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             "The leaking model substantially outperforms the correct one in macro-F1 "
             "(0.757–0.857 vs 0.323–0.410), yet is rejected by the verification criterion "
             "(V = 0) for every algorithm.",
             ha="center", fontsize=7.0, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig1_leakage")


# ═══════════════════════ Fig. 2 — component heat map ═══════════════════════
def fig2():
    df = pd.read_csv(OUT / "final_table.csv").set_index("config")
    rows = [("M0 Корректная", "M0 — reference model", None),
            ("D-T тихая утечка", "D-T — data leakage", "T"),
            ("D-F ложное объяснение", "D-F — false explanation", "F"),
            ("D-S нестабильная", "D-S — instability", "S"),
            ("D-C инверсия логики", "D-C — domain-logic inversion", "C"),
            ("D-R неопределённость", "D-R — low certainty", "R")]
    cols = ["T", "F", "S", "C", "R", "V"]
    titles = ["T\ntemporal\nadmissibility", "F\nexplanation\nfidelity", "S\nstability",
              "C\ndomain\nconsistency", "R\nreliability", "V\nintegral\nindex"]
    M = np.array([[df.loc[k, c] for c in cols] for k, _, _ in rows], float)

    fig, ax = plt.subplots(figsize=(6.69, 3.95))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i, (_, _, tgt) in enumerate(rows):
        for j, c in enumerate(cols):
            v, hit = M[i, j], (c == tgt)
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8.1 if hit else 11.5,
                    color="white" if v < 0.35 else "#111",
                    fontweight="bold" if (hit or c == "V") else "normal")
            if hit:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           ec="#B71C1C", lw=2.04, zorder=6))
    ax.axvline(len(cols) - 1.5, color=DARK, lw=1.77, zorder=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(titles, fontsize=7.0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l, _ in rows], fontsize=7.0)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylim(len(rows) - 0.35, -0.65); ax.set_xlim(-0.65, len(cols) - 0.35)
    x0, x1 = -0.5, len(cols) - 0.5
    for xs, ys in (((x0, x1), (-.5, -.5)), ((x0, x1), (.5, .5)),
                   ((x0, x0), (-.5, .5)), ((x1, x1), (-.5, .5))):
        ax.plot(xs, ys, color="#1B5E20", lw=1.9, zorder=8,
                solid_capstyle="projecting", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Component value", fontsize=7.0); cb.outline.set_visible(False)
    ax.set_title("Heat map of verification components",
                 fontsize=8.7, fontweight="bold", loc="left", pad=16)
    fig.text(0.5, -0.055,
             "Each defect reduces exactly its own component (outlined in red) while the remaining "
             "components stay in the normal range — the defects are isolated.",
             ha="center", fontsize=7.0, style="italic", color="#333")
    fig.text(0.5, -0.105,
             f"Integral index: reference model {M[0,5]:.3f} — verified; all defective configurations "
             f"rejected at threshold θ = {THETA1}. Leakage nullifies V unconditionally through the "
             "veto on component T.",
             ha="center", fontsize=7.0, color="#444")
    save(fig, "fig2_components_heatmap")


# ═══════════════════════ Fig. 3 — significance ═══════════════════════
def fig3(res, rng):
    y = res["defective"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 3.25))
    for col, name, c, w in (("V", "Verifiability index V", BLUE, 2.8),
                            ("macro_f1", "Macro-F1", RED, 2.4)):
        fpr, tpr, _ = roc_curve(y, 1 - res[col].values)
        auc = roc_auc_score(y, 1 - res[col].values)
        a1.plot(fpr, tpr, lw=w, color=c, label=f"{name} — AUROC {auc:.3f}", zorder=3)
        a1.fill_between(fpr, tpr, alpha=0.07, color=c, zorder=1)
    a1.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.09, label="random guessing", zorder=2)
    a1.set_xlabel("False positive rate", fontsize=7.0)
    a1.set_ylabel("True positive rate", fontsize=7.0)
    a1.set_title("A. ROC curves for defect detection", fontsize=7.8,
                 fontweight="bold", loc="left")
    a1.legend(loc="lower right", fontsize=7.0, framealpha=0.96)
    a1.set_xlim(-0.02, 1.02); a1.set_ylim(-0.02, 1.04)
    clean(a1)

    groups = {k: g.index.values for k, g in res.groupby("config")}
    keys = list(groups); diffs = []
    for _ in range(3000):
        idx = np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys), True)])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], 1 - res["V"].values[idx])
                     - roc_auc_score(y[idx], 1 - res["macro_f1"].values[idx]))
    diffs = np.array(diffs)

    # ВАЖНО: границы доверительного интервала берутся из stress2_summary.json —
    # того же файла, откуда они попадают в текст статьи. Раньше рисунок
    # пересчитывал бутстрэп своим генератором и показывал [0.101; 0.389],
    # тогда как в тексте стояло [0.101; 0.391]: два независимых прогона
    # одного случайного процесса расходятся в третьем знаке. Единый источник
    # истины исключает такое расхождение.
    with open(OUT / "stress2_summary.json", encoding="utf-8") as fh:
        lo, hi = json.load(fh)["CI95_разницы_кластерный"]

    a2.hist(diffs, bins=46, color=BLUE, alpha=0.80, edgecolor="white", lw=0.34, zorder=3)
    a2.axvspan(lo, hi, color=BLUE, alpha=0.11, zorder=1)
    for v in (lo, hi):
        a2.axvline(v, ls="--", color=DARK, lw=1.02, zorder=4)
    a2.axvline(0, color=RED, lw=1.77, zorder=5)
    a2.annotate("zero", (0, a2.get_ylim()[1] * 0.96), color=RED, fontsize=7.0,
                fontweight="bold", ha="left", xytext=(6, 0), textcoords="offset points")
    a2.set_xlabel("AUROC(V) − AUROC(macro-F1)", fontsize=7.0)
    a2.set_ylabel("Bootstrap resamples", fontsize=7.0)
    a2.set_title("B. Cluster bootstrap of the AUROC difference", fontsize=7.8,
                 fontweight="bold", loc="left")
    a2.legend(handles=[Patch(facecolor=BLUE, alpha=.45, label=f"95% CI [{lo:.3f}; {hi:.3f}]"),
                       Patch(facecolor=RED, label="zero lies outside the interval")],
              fontsize=7.0, loc="upper right", framealpha=0.96)
    clean(a2)
    fig.suptitle("Statistical significance of the advantage of index V",
                 fontsize=8.4, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             f"In all 3000 cluster bootstrap resamples the difference is positive; "
             f"the confidence interval [{lo:.3f}; {hi:.3f}] excludes zero.",
             ha="center", fontsize=7.0, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig3_significance")


# ═══════════════════════ Fig. 4 — separation ═══════════════════════
def fig4(res):
    order = ["R", "C", "S", "F", "T", "D", "—"]
    names = {"—": "REFERENCE\n(correct)", "D": "corrupted\ndata", "T": "data leakage",
             "F": "false\nexplanation", "S": "instability",
             "C": "logic\ninversion", "R": "input noise"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 3.50), sharey=True)
    rng = np.random.default_rng(7)
    for ax, col, title, xlab in ((a1, "macro_f1", "A. Predictive accuracy (macro-F1)", "Macro-F1"),
                                 (a2, "V", "B. Verifiability index V", "V")):
        for yi, t in enumerate(order):
            sub = res[res["тип"] == t]
            if not len(sub):
                continue
            c = GREEN if t == "—" else RED
            ax.scatter(sub[col], yi + rng.uniform(-.20, .20, len(sub)), s=12, color=c,
                       alpha=.50, edgecolor="white", linewidth=.5, zorder=3)
            m = sub[col].mean()
            ax.plot([m, m], [yi - .34, yi + .34], color=DARK, lw=1.77, zorder=4)
            ax.annotate(f"{m:.2f}", (m, yi + .44), ha="center", fontsize=7.0,
                        fontweight="bold", color=DARK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[t] for t in order], fontsize=7.0)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.75, len(order) - 0.15)
        ax.set_xlabel(xlab, fontsize=7.0)
        ax.set_title(title, fontsize=7.8, fontweight="bold", loc="left")
        ei = order.index("—")
        ax.axhspan(ei - .5, ei + .5, color=GREEN, alpha=.09, zorder=0)
        clean(ax)
    a2.axvline(THETA2, ls="--", color=DARK, lw=1.22, zorder=2,
               label=f"threshold θ = {THETA2}")
    a2.legend(loc="lower right", fontsize=7.0, framealpha=0.96)
    fig.legend(handles=[Patch(facecolor=GREEN, alpha=.6, label="correct configurations (n = 12)"),
                        Patch(facecolor=RED, alpha=.6, label="defective configurations (n = 336)")],
               loc="lower center", ncol=2, fontsize=7.0, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Separation of correct and defective configurations (n = 348)",
                 fontsize=8.4, fontweight="bold", y=1.0)
    fig.text(0.5, -0.115,
             "Each point is one configuration; vertical bars are group means. Accuracy does not "
             "separate the reference model from defective ones; index V does.",
             ha="center", fontsize=7.0, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig4_separation")


# ═══════════════════════ Fig. 5 — intensity ═══════════════════════
def fig5(res):
    v0 = float(res.loc[res.defective == 0, "V"].mean())
    v_best = float(res.loc[res.defective == 0, "V"].max())
    spec = [("T", "Data leakage", "#C62828", "o", 3.0),
            ("F", "False explanation", "#7B1FA2", "s", 2.4),
            ("S", "Instability", "#00838F", "^", 2.4),
            ("C", "Domain-logic inversion", "#EF6C00", "D", 2.4),
            ("R", "Low certainty", "#546E7A", "v", 2.4),
            ("D", "Data corruption", "#B71C1C", "X", 3.0)]
    fig, ax = plt.subplots(figsize=(6.69, 4.15))
    for t, label, col, mk, lw in spec:
        sub = (res[(res["тип"] == t) & (res.defective == 1)]
               .groupby("уровень")["V"].mean().sort_index())
        if not len(sub):
            continue
        lv = sub.index.values.astype(float)
        x = np.concatenate([[0.0], lv / lv.max()])
        yv = np.concatenate([[v0], sub.values])
        ax.plot(x, yv, mk + "-", color=col, lw=lw, ms=5.0, mec="white", mew=1.4,
                label=label, zorder=4, alpha=.95)
    ax.axhline(THETA2, ls="--", color=DARK, lw=1.36, zorder=3)
    ax.text(1.005, THETA2, f"  verification threshold\n  V = θ = {THETA2}",
            fontsize=7.0, color=DARK, fontweight="bold", va="center")
    ax.axhspan(THETA2, 1.0, color=GREEN, alpha=.055, zorder=0)
    ax.axhspan(0.0, THETA2, color=RED, alpha=.045, zorder=0)
    ax.text(.985, .845, "VERIFICATION ZONE", fontsize=7.0, color=GREEN,
            fontweight="bold", alpha=.9, ha="right")
    ax.text(.985, .035, "REJECTION ZONE", fontsize=7.0, color=RED,
            fontweight="bold", alpha=.9, ha="right")
    ax.plot([0], [v_best], "*", ms=9.3, mfc=GREEN, mec="white", mew=1.2, zorder=7)
    ax.plot([0], [v0], "o", ms=7.4, mfc="white", mec=DARK, mew=2.4, zorder=7)
    ax.set_xlabel("Defect intensity\n(0 — no defect, 1 — maximum level tested)", fontsize=7.1)
    ax.set_ylabel("Verifiability index V", fontsize=7.1)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.035, 0.90)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, fontsize=7.0,
              framealpha=0, title="Defect type", title_fontsize=7.0)
    clean(ax)
    ax.set_title("Verifiability index as a function of methodological defect intensity",
                 fontsize=8.4, fontweight="bold", loc="left", pad=14)
    fig.text(0.5, -0.245,
             "Data leakage (T) and data corruption (D) nullify the index already at minimum "
             "intensity — the veto is triggered.\nFalse explanation (F) and domain-logic inversion "
             "(C) reduce the index monotonically.",
             ha="center", fontsize=7.0, style="italic", color="#333")
    fig.text(0.5, -0.325,
             "Intensity levels of different defects are defined in different units (ρ, η, sample "
             "fraction, noise multiplier), therefore\nintensity is rescaled as a fraction of the "
             "maximum level tested.",
             ha="center", fontsize=7.0, color="#666")
    save(fig, "fig5_intensity")


# ═══════════════════════ Fig. 6 — monotonicity ═══════════════════════
def fig6(res):
    spec = [("F", "False explanation", "#7B1FA2", "decreasing"),
            ("C", "Domain-logic inversion", "#EF6C00", "decreasing"),
            ("S", "Instability", "#00838F", "increases with sample size"),
            ("R", "Input noise", GREY, "no response")]
    fig, axes = plt.subplots(1, 4, figsize=(6.69, 2.75), sharey=True)
    for ax, (t, title, col, note) in zip(axes, spec):
        sub = (res[(res["тип"] == t) & (res.defective == 1)]
               .groupby("уровень")["V"].mean())
        x = np.arange(len(sub))
        ax.plot(x, sub.values, "o-", color=col, lw=1.63, ms=5.0, mec="white", mew=1.5, zorder=3)
        ax.fill_between(x, 0, sub.values, color=col, alpha=.10, zorder=1)
        ax.axhline(THETA2, ls="--", color="#555", lw=0.82, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in sub.index], fontsize=7.0)
        ax.set_title(f"{title}\n({t}) — {note}", fontsize=7.0, fontweight="bold",
                     color=col, pad=9)
        ax.set_xlabel("defect level", fontsize=7.0)
        ax.set_ylim(0, .92)
        clean(ax)
        if sub.values.max() - sub.values.min() > 0.10:      # только выраженная динамика
            for i, v in enumerate(sub.values):
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                            xytext=(0, 13), ha="center",
                            fontsize=7.0, color=col, fontweight="bold")
        else:                                                # плоская кривая: только концы
            for i in (0, len(sub) - 1):
                ax.annotate(f"{sub.values[i]:.3f}", (i, sub.values[i]),
                            textcoords="offset points", xytext=(0, 13), ha="center",
                            fontsize=7.0, color=col, fontweight="bold")
    axes[0].set_ylabel("Verifiability index V", fontsize=7.0)
    fig.text(0.5, -0.045,
             "Defects D (data corruption) and T (data leakage): V = 0 at every intensity level — "
             "the veto is unconditional.",
             ha="center", fontsize=7.0, color=RED, fontweight="bold")
    fig.suptitle(f"Behaviour of index V under increasing defect intensity, by defect type (dashed line — threshold θ = {THETA2})",
                 fontsize=8.4, fontweight="bold", y=1.06)
    fig.text(0.5, -0.10,
             "The index discriminates not only the presence of a violation but also its severity. "
             "Component R is insensitive to its own defect in the current implementation.",
             ha="center", fontsize=7.0, style="italic", color="#444")
    fig.tight_layout()
    save(fig, "fig6_monotonicity")


# ═══════════════════════ Fig. 7 — ablation ═══════════════════════
def fig7():
    abl = pd.read_csv(OUT / "stress2_ablation.csv")
    tr = {"полная V": "Full formula V", "без D": "without D", "без T": "without T",
          "без F": "without F", "без S": "without S", "без C": "without C",
          "без R": "without R", "арифм. среднее": "Arithmetic mean",
          "только Macro-F1": "Macro-F1 only"}
    labels = [tr.get(v, v) for v in abl["вариант"]]
    cols = ["D", "T", "F", "S", "C", "R"]
    M = abl[cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(6.69, 4.35))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.1, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.0,
                    color="white" if v < .40 else "#111",
                    fontweight="bold" if v < .68 else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"defect {c}" for c in cols], fontsize=7.0)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylim(len(labels) - .35, -.65); ax.set_xlim(-.65, len(cols) - .35)
    for i, lab in enumerate(labels):
        ec = "#1B5E20" if lab.startswith("Full") else ("#B71C1C" if "Macro-F1" in lab else None)
        if not ec:
            continue
        x0, x1, y0, y1 = -.5, len(cols) - .5, i - .5, i + .5
        for xs, ys in (((x0, x1), (y0, y0)), ((x0, x1), (y1, y1)),
                       ((x0, x0), (y0, y1)), ((x1, x1), (y0, y1))):
            ax.plot(xs, ys, color=ec, lw=1.9, zorder=7,
                    solid_capstyle="projecting", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=.030, pad=.02)
    cb.set_label("AUROC of defect detection", fontsize=7.0); cb.outline.set_visible(False)
    ax.set_title("Ablation: contribution of each component to defect detection",
                 fontsize=8.1, fontweight="bold", loc="left", pad=16)
    fig.text(0.5, -0.02,
             "Removing a component collapses the detection of its corresponding defect. Bottom row: "
             "macro-F1 yields AUROC 0.111 for leakage — systematically worse than random guessing.",
             ha="center", fontsize=7.0, style="italic", color="#444")
    save(fig, "fig7_ablation")


# ═══════════════════════ Fig. 8 — semi-synthetic ═══════════════════════
def fig8():
    d = pd.read_csv(OUT / "semisynth_recovery.csv").set_index("алгоритм")
    algos = ["LogReg", "RandomForest", "HistGB"]
    names = {"LogReg": "Logistic\nregression", "RandomForest": "Random\nforest",
             "HistGB": "Gradient\nboosting"}
    x = np.arange(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 3.35))

    f1 = [d.loc[a, "macro_f1_synth"] for a in algos]
    bars = a1.bar(x, f1, .52, color=BLUE, edgecolor="white", lw=0.82)
    a1.bar_label(bars, labels=[f"{v:.3f}" for v in f1], padding=4, fontsize=7.1,
                 fontweight="bold")
    a1.set_ylim(0.78, 0.845)
    a1.set_ylabel("Macro-F1", fontsize=7.0)
    a1.set_title("A. Predictive accuracy", fontsize=7.8, fontweight="bold", loc="left")
    a1.axhspan(min(f1), max(f1), color=RED, alpha=.09, zorder=0)
    a1.axhline(min(f1), ls=":", color=RED, lw=0.88, zorder=2)
    a1.axhline(max(f1), ls=":", color=RED, lw=0.88, zorder=2)
    a1.text(1.0, .838, f"total accuracy spread — {max(f1) - min(f1):.3f}",
            fontsize=7.0, color=RED, fontweight="bold", ha="center")
    a1.text(.02, .02, "axis truncated", fontsize=7.0, color="#888", style="italic",
            transform=a1.transAxes)

    w = .30
    p3 = [d.loc[a, "Precision@3"] for a in algos]
    r3 = [d.loc[a, "Recall@3"] for a in algos]
    sa = [d.loc[a, "SignAccuracy"] for a in algos]
    b1 = a2.bar(x - w, p3, w, label="Precision@3", color=GREEN, edgecolor="white", lw=0.82)
    b2 = a2.bar(x, r3, w, label="Recall@3", color="#66BB6A", edgecolor="white", lw=0.82)
    b3 = a2.bar(x + w, sa, w, label="Sign accuracy", color="#B0BEC5", edgecolor="white", lw=0.82)
    a2.bar_label(b1, labels=[f"{v:.2f}" for v in p3], padding=3, fontsize=7.0,
                 fontweight="bold")
    a2.set_ylim(0, 1.12)
    a2.set_ylabel("Metric value", fontsize=7.0)
    a2.set_title("B. Recovery of the ground-truth explanation", fontsize=7.8,
                 fontweight="bold", loc="left")
    a2.legend(fontsize=7.0, loc="upper right", framealpha=.96)

    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels([names[a] for a in algos], fontsize=7.0)
        clean(ax)
    fig.suptitle("Predictive accuracy and ground-truth explanation recovery on semi-synthetic data",
                 fontsize=8.4, fontweight="bold", y=1.0)
    fig.text(0.5, -0.055,
             "At virtually identical accuracy (spread 0.011) the quality of explanation differs "
             "fundamentally: logistic regression recovers the planted mechanism completely "
             "(Precision@3 = Recall@3 = 1.00),\nwhereas tree-based models admit noise features into "
             "the explanation (0.33). Sign accuracy equals 1.00 for all algorithms.",
             ha="center", fontsize=7.0, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig8_semisynthetic")


# ═══════════════════════ Fig. 9 — all indicators ═══════════════════════
def fig9(res, rng):
    y = res["defective"].values
    res = res.copy()
    res["confidence"] = 1.0 / res["conf_size"].clip(lower=1.0)
    series = [("V", "Integral index V", BLUE, 3.4, "-"),
              ("macro_f1", "Macro-F1", RED, 2.6, "-"),
              ("confidence", "Model confidence", "#8D6E63", 2.0, "-."),
              ("F", "Component F (explanation)", "#7B1FA2", 1.8, "--"),
              ("S", "Component S (stability)", "#00838F", 1.8, "--"),
              ("C", "Component C (consistency)", "#EF6C00", 1.8, "--"),
              ("R", "Component R (reliability)", "#546E7A", 1.8, "--")]

    groups = {k: g.index.values for k, g in res.groupby("config")}
    keys = list(groups)
    boots = {c: [] for c, *_ in series}
    for _ in range(2000):
        idx = np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys), True)])
        if len(np.unique(y[idx])) < 2:
            continue
        for c, *_ in series:
            try:
                boots[c].append(roc_auc_score(y[idx], 1 - res[c].values[idx]))
            except Exception:
                pass

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 3.90),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    rows = []
    for col, label, c, lw, ls in series:
        fpr, tpr, _ = roc_curve(y, 1 - res[col].values)
        auc = roc_auc_score(y, 1 - res[col].values)
        lo, hi = np.percentile(np.array(boots[col]), [2.5, 97.5])
        main = col in ("V", "macro_f1")
        a1.plot(fpr, tpr, ls, color=c, lw=lw if main else 1.4,
                alpha=1.0 if main else .42, zorder=5 if main else 3,
                label=f"{label} — {auc:.3f}" if main else None)
        rows.append((label, auc, lo, hi, c))
    a1.plot([0, 1], [0, 1], ":", color="#9E9E9E", lw=1.02,
            label="random guessing — 0.500", zorder=2)
    a1.plot([], [], "-", color="#9E9E9E", lw=0.95, alpha=.6,
            label="individual components (5 curves)")
    a1.set_xlabel("False positive rate", fontsize=7.0)
    a1.set_ylabel("True positive rate", fontsize=7.0)
    a1.set_title("A. ROC curves for detecting defective models",
                 fontsize=7.8, fontweight="bold", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.0, framealpha=0)
    a1.set_xlim(-.02, 1.02); a1.set_ylim(-.02, 1.03)
    clean(a1)

    rows.sort(key=lambda r: r[1])
    for i, (label, auc, lo, hi, c) in enumerate(rows):
        a2.plot([lo, hi], [i, i], color=c, lw=2.04, solid_capstyle="round", zorder=3)
        a2.plot([auc], [i], "o", ms=6.2, color=c, mec="white", mew=1.6, zorder=4)
        a2.annotate(f"{auc:.3f}  [{lo:.3f}; {hi:.3f}]", (hi, i), xytext=(10, 0),
                    textcoords="offset points", fontsize=7.0, va="center",
                    color=c, fontweight="bold")
    a2.axvline(.5, ls=":", color="#9E9E9E", lw=1.09)
    a2.text(.5, len(rows) - .42, "random\nguessing", fontsize=7.0, color="#757575",
            ha="center", va="top")
    iv = next(i for i, r in enumerate(rows) if r[0].startswith("Integral"))
    im = next(i for i, r in enumerate(rows) if r[0] == "Macro-F1")
    a2.axvline(rows[iv][2], ls="--", color="#1B5E20", lw=0.88, alpha=.75, zorder=1)
    a2.text(rows[iv][2], -1.15,
            f"lower bound of V ({rows[iv][2]:.3f}) exceeds upper bound of macro-F1 "
            f"({rows[im][3]:.3f})",
            fontsize=7.0, color="#1B5E20", fontweight="bold", ha="center")
    a2.set_yticks(np.arange(len(rows)))
    a2.set_yticklabels([r[0] for r in rows], fontsize=7.0)
    a2.set_xlabel("AUROC (95% confidence interval)", fontsize=7.0)
    a2.set_title("B. AUROC with confidence intervals", fontsize=7.8,
                 fontweight="bold", loc="left")
    a2.set_xlim(.05, 1.16); a2.set_ylim(-1.5, len(rows) - .2)
    clean(a2)
    fig.suptitle("Ability of different indicators to detect methodologically defective models "
                 "(348 configurations)", fontsize=8.4, fontweight="bold", y=1.0)
    fig.text(0.5, -0.045,
             "The integral index V outperforms both macro-F1 and every individual component — "
             "the discriminative ability arises precisely from the joint non-compensatory "
             "aggregation. Confidence intervals were obtained by cluster bootstrap over configurations.",
             ha="center", fontsize=7.0, style="italic", color="#333")
    fig.tight_layout()
    save(fig, "fig9_all_indicators")


def main():
    rng = np.random.default_rng(20260802)
    res = pd.read_csv(OUT / "stress2_all.csv")
    print(f"loaded: {len(res)} configurations\n")
    fig1(); fig2(); fig3(res, rng); fig4(res); fig5(res)
    fig6(res); fig7(); fig8(); fig9(res, rng)
    print(f"\nall figures saved to {EN}")


if __name__ == "__main__":
    main()
