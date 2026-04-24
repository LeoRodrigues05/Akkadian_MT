"""
STEP 18 — Generate report figures.

Produces, into ``documentation/images/``:

  * fig_results_comparison.pdf   -- grouped bar chart: BLEU / chrF++ / Kaggle
                                    public score for the five compared systems.
  * fig_corpus_composition.pdf   -- horizontal stacked bar: source breakdown of
                                    augmented_train.csv (base / OCR / drills /
                                    template / pseudo-label).

Run from the repo root after the augmented training CSVs and eval JSONs
exist on disk:

    python scripts/18_make_figures.py
"""
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "documentation", "images")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Figure 1: results comparison ──────────────────────────────────────────
def fig_results_comparison():
    """Grouped bars: BLEU / chrF++ / Kaggle public per system."""
    systems = OrderedDict([
        # label                        BLEU    chrF++   Kaggle pub
        ("Transformer\n(augmented)",   (14.55, 33.12,  12.71)),
        ("BiLSTM\n(augmented)",        (33.58, 49.51,  18.94)),
        ("ByT5-base\n(aligned)",       (19.46, 37.71,  25.96)),
        ("ByT5-base SFT\n(augmented)", (33.27, 49.75,  29.41)),
        ("ByT5-large SFT\n(augmented)",(31.82, 49.96,  np.nan)),
    ])
    metrics = ["BLEU (val)", "chrF++ (val)", "chrF++ (Kaggle pub.)"]
    colors  = ["#4C72B0", "#55A868", "#C44E52"]

    n_systems = len(systems)
    n_metrics = len(metrics)
    x = np.arange(n_systems)
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for i, (m_name, color) in enumerate(zip(metrics, colors)):
        vals = [v[i] for v in systems.values()]
        offsets = x + (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(offsets, [0 if np.isnan(v) else v for v in vals],
                      width=width, color=color, label=m_name,
                      edgecolor="white", linewidth=0.5)
        for off, v, bar in zip(offsets, vals, bars):
            if np.isnan(v):
                ax.text(off, 1.0, "n/a", ha="center", va="bottom",
                        fontsize=8, color="grey")
            else:
                ax.text(off, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(systems.keys()), fontsize=9)
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 60)
    ax.set_title("Validation vs Kaggle scores by system")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    out = os.path.join(OUT_DIR, "fig_results_comparison.pdf")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 2: augmented corpus composition ────────────────────────────────
def fig_corpus_composition():
    """Donut chart: row counts in augmented_train.csv by `source`."""
    csv_path = os.path.join(DATA_DIR, "augmented_train.csv")
    if not os.path.exists(csv_path):
        print(f"  skipping corpus composition: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    if "source" not in df.columns:
        print("  skipping corpus composition: no `source` column")
        return

    counts = df["source"].value_counts()
    order = ["base", "ocr", "drills", "template", "pseudo"]
    display = {
        "base":     "Aligned base pairs",
        "ocr":      "OCR-extracted pairs",
        "drills":   "Lexicon drills",
        "template": "Template patterns",
        "pseudo":   "Pseudo-labelled monolingual",
    }
    palette = {
        "base":     "#4C72B0",
        "ocr":      "#8172B2",
        "drills":   "#CCB974",
        "template": "#64B5CD",
        "pseudo":   "#55A868",
    }

    sources = [s for s in order if s in counts.index]
    values = [int(counts[s]) for s in sources]
    colors = [palette[s] for s in sources]
    total = sum(values)
    legend_labels = [
        f"{display[s]} — {v:,} ({100*v/total:.1f}%)"
        for s, v in zip(sources, values)
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
    )
    # Annotate large slices in-place; small ones go to the legend only.
    for w, v in zip(wedges, values):
        frac = v / total
        if frac < 0.05:
            continue
        ang = (w.theta2 + w.theta1) / 2.0
        r = 0.78  # mid-ring
        x = r * np.cos(np.deg2rad(ang))
        y = r * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{frac*100:.1f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_aspect("equal")
    ax.set_title(f"Composition of the augmented training corpus "
                 f"(total: {total:,} pairs)", pad=12)
    ax.legend(
        wedges, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        borderaxespad=0,
    )

    out = os.path.join(OUT_DIR, "fig_corpus_composition.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print(f"Writing figures to: {OUT_DIR}")
    fig_results_comparison()
    fig_corpus_composition()


if __name__ == "__main__":
    main()
