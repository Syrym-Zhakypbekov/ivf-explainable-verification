"""Fig. 1 — architecture of explainable-verification orchestration (EN).

Redrawn 07.09.2026 after the reviewer's remark: component R is "Conformal informativeness"
(not "Conformal reliability"). Pure matplotlib, no data needed. Output: figures/en/fig0_architecture.{png,pdf}
Run:  python src/21_fig_architecture_en.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "en"
OUT.mkdir(parents=True, exist_ok=True)

TOP = [("Input data\nand metadata", "#E3EEF8", "#3E5F80"),
       ("Data\nadmissibility D", "#FBE3E3", "#B23A3A"),
       ("Temporal\nadmissibility T", "#FBE3E3", "#B23A3A"),
       ("Predictive model\nand explanation", "#E5F3E5", "#2E7D32"),
       ("Verification\norchestrator", "#FDEFD8", "#C77800"),
       ("Verdict and\naudit record", "#E3EEF8", "#3E5F80")]
AGENTS = [("F\nExplanation\nfidelity"), ("S\nExplanation\nstability"),
          ("C\nDomain\nconsistency"), ("R\nConformal\ninformativeness")]


def box(ax, x, y, w, h, text, fc, ec, fs=9.2, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#222", linespacing=1.35)


def arrow(ax, p, q, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color="#444", shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    ax.set_xlim(0, 20.2); ax.set_ylim(0, 9.4); ax.axis("off")

    ax.text(0.2, 9.05, "Explainable verification architecture", fontsize=13, fontweight="bold", va="center")

    # top row
    w, h, gap, y0 = 2.75, 1.15, 0.45, 7.2
    xs = [0.3 + i * (w + gap) for i in range(6)]
    for x, (t, fc, ec) in zip(xs, TOP):
        box(ax, x, y0, w, h, t, fc, ec)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y0 + h / 2), (b, y0 + h / 2))

    # agents row
    aw, ah, y1 = 3.3, 1.45, 3.75
    axs = [1.2 + i * (aw + 1.3) for i in range(4)]
    orch = (xs[4] + w / 2, y0)
    for x, t in zip(axs, AGENTS):
        box(ax, x, y1, aw, ah, t, "#F4F4F4", "#5A6B7C")
        arrow(ax, orch, (x + aw / 2, y1 + ah))

    # aggregation box
    gw, gh, gx, gy = 7.6, 1.55, 6.2, 0.75
    box(ax, gx, gy, gw, gh,
        "Non-compensatory aggregation\nV = D · T · (F · S · C · R)$^{1/4}$", "#FDF6DC", "#A68A00", fs=10.5)
    for x in axs:
        arrow(ax, (x + aw / 2, y1), (gx + gw / 2 + (x + aw / 2 - 10) * 0.12, gy + gh))
    # verdict feedback: aggregation -> right margin -> up -> into the verdict box (no crossing of R)
    xr = xs[5] + w + 0.55
    ax.plot([gx + gw, xr, xr], [gy + gh / 2, gy + gh / 2, y0 + h / 2], color="#444", lw=1.4, solid_capstyle="round")
    arrow(ax, (xr, y0 + h / 2), (xs[5] + w, y0 + h / 2))

    ax.text(0.2, 0.18, "D and T are veto gates; F, S, C and R are independently reproducible white-box verification agents.",
            fontsize=8.6, style="italic", color="#333", va="center")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_architecture.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("saved:", OUT / "fig0_architecture.png")


if __name__ == "__main__":
    main()
