"""Paper figures, drawn from the result files in the repository (nothing typed by hand).

    python paper/icml/figures/make_figures.py        # from the repository root

fig_teaser.pdf  browser step success (Mind2Web test) against out-of-domain general accuracy (transfer-v4 test)
fig_done.pdf    DONE recall against premature-DONE rate as the DONE threshold is swept (NNetNav test)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOCKED, BASE, CURVES = ROOT / "results/locked", ROOT / "results/baselines", ROOT / "paper/icml/data"

# categorical slots 1-3 of the reference palette (validated: CVD dE 9.2, normal-vision dE 27.6); the aqua slot is
# below 3:1 contrast on white, so every mark carries a direct text label
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = {"1.7b": "#86b6ef", "4b": "#2a78d6", "8b": "#184f95"}   # ordinal sizes: one hue, light to dark
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False,
})


def read(path):
    return json.loads(Path(path).read_text())


def teaser():
    n_all = read(BASE / "kev-4b-m2w-v2.json")["n"]
    pts = []
    for m in ("1.7b", "4b", "8b"):
        m2w = read(LOCKED / f"wev-{m}/m2w-v2.json")
        step = m2w["step_success"] * m2w["n"] / n_all   # requests beyond wev's context count as wrong
        gen = read(LOCKED / f"wev-{m}/kev-transfer-v4.json")["all_questions"]["accuracy"]
        pts.append(("wev", f"wev-{m}", gen, step))
    for key, label, fam in (("kev-4b", "Kev-4B", "Kev"), ("kev-8b", "Kev-8B", "Kev"),
                            ("laya-typed-decisions", "Laya (TD)", "Laya"), ("laya", "Laya", "Laya")):
        pts.append((fam, label, read(BASE / f"{key}-kev-transfer-v4.json")["all_questions"]["accuracy"],
                    read(BASE / f"{key}-m2w-v2.json")["step_success"]))
    color = {"wev": BLUE, "Kev": ORANGE, "Laya": AQUA}
    offset = {"wev-1.7b": (-4, -9, "right"), "wev-4b": (-5, 4, "right"), "wev-8b": (5, -9, "left"),
              "Kev-4B": (-5, 5, "right"), "Kev-8B": (-5, 5, "right"), "Laya (TD)": (-5, 7, "right"),
              "Laya": (6, 4, "left")}

    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for fam, label, x, y in pts:
        ax.scatter(100 * x, 100 * y, s=46, color=color[fam], edgecolor="white", linewidth=1.2, zorder=3,
                   marker="o" if fam == "wev" else ("s" if fam == "Kev" else "^"))
        dx, dy, ha = offset[label]
        ax.annotate(label, (100 * x, 100 * y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7, color=INK, fontweight="bold" if fam == "wev" else "normal")
    ax.set_xlim(58, 86)
    ax.set_ylim(-6, 90)
    ax.set_xlabel("General decisions, out of domain (transfer-v4 acc., %)")
    ax.set_ylabel("Browser steps, unseen sites\n(Mind2Web step success, %)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=mk, color=c, linestyle="", markersize=6, markeredgecolor="white", label=l)
               for mk, c, l in (("o", BLUE, "wev (ours)"), ("s", ORANGE, "Kev"), ("^", AQUA, "Laya"))]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=7, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_teaser.pdf")


def done_curve():
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    for m in ("1.7b", "4b", "8b"):
        c = read(CURVES / f"done-curve-wev-{m}.json")
        keep = [i for i, t in enumerate(c["thresholds"]) if t >= 0.05]
        x = [100 * c["premature"][i] for i in keep]
        y = [100 * c["recall"][i] for i in keep]
        ax.plot(x, y, color=BLUE_RAMP[m], linewidth=2 if m == "4b" else 1.4, label=f"wev-{m}", zorder=2)
        for t, mk in ((0.5, "o"), (0.8, "s")):
            i = c["thresholds"].index(t)
            ax.scatter(100 * c["premature"][i], 100 * c["recall"][i], s=26, marker=mk, color=BLUE_RAMP[m],
                       edgecolor="white", linewidth=1.0, zorder=3)
            if m == "4b":
                ax.annotate(f"threshold {t}", (100 * c["premature"][i], 100 * c["recall"][i]),
                            xytext=(4, -11), textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, 16)
    ax.set_ylim(40, 95)
    ax.set_xlabel("Premature DONE (% of not-finished steps)")
    ax.set_ylabel("DONE recall (%)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_done.pdf")


if __name__ == "__main__":
    teaser()
    done_curve()
    print("wrote", OUT / "fig_teaser.pdf", OUT / "fig_done.pdf")
