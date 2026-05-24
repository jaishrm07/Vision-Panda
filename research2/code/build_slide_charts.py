"""Render the three slide PNG charts the video deck needs.

Outputs at 1920x1080 into research2/results/slide_charts/:
  - axis_grouped_bars.png      (slide 6: OOD axis ranking, budget 50, 3 families)
  - threshold_sweep.png        (slide 7: OOD success vs precision threshold, budget 200)
  - family_precision_end.png   (slide 8: precision @1cm vs end-distance, budget 200)
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis_id_ood_all_budgets"
OUT_DIR = ROOT / "results" / "slide_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMS = ["Scratch CNN", "Frozen ResNet-18", "Partial ResNet-18"]
FAM_COLOR = {
    "Scratch CNN": "#2b4f81",
    "Frozen ResNet-18": "#2f7a4d",
    "Partial ResNet-18": "#c46a1d",
}

DPI = 160
W_IN, H_IN = 12.0, 6.75  # 1920x1080 at 160dpi


def load(name):
    with open(ANALYSIS / name) as f:
        return list(csv.DictReader(f))


def num(row, k):
    v = row.get(k)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax.yaxis.grid(True, color="#e6e3dc", lw=0.8)
    ax.set_axisbelow(True)


def chart_axis_bars():
    rows = [r for r in load("ood_axis_by_family_budget.csv") if int(r["budget"]) == 50]
    axes_order = ["Color", "Camera viewpoint", "Lighting", "Spatial distribution"]
    data = {f: [] for f in FAMS}
    for ax_name in axes_order:
        for f in FAMS:
            r = next((r for r in rows if r["family_label"] == f and r["axis_label"] == ax_name), None)
            data[f].append(num(r, "success_at_1cm_pct") if r else 0.0)

    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    x = np.arange(len(axes_order))
    width = 0.26
    for i, f in enumerate(FAMS):
        offs = (i - 1) * width
        bars = ax.bar(x + offs, data[f], width, label=f, color=FAM_COLOR[f], edgecolor="white", lw=1.2)
        for bar, val in zip(bars, data[f]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=11, color="#222")
    ax.set_title("OOD success@1cm by visual axis  ·  budget 50, 3 seeds", fontsize=18, weight="bold", pad=14)
    ax.set_ylabel("OOD success@1cm (%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(axes_order, fontsize=13)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.18)
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = OUT_DIR / "axis_grouped_bars.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_threshold_sweep():
    rows = [r for r in load("overall_by_family_split_budget.csv")
            if r["split"] == "ood" and int(r["budget"]) == 200]
    thresholds = ["1cm", "2cm", "5cm"]
    keys = ["success_at_1cm_pct", "success_at_2cm_pct", "success_at_5cm_pct"]

    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    for f in FAMS:
        r = next((r for r in rows if r["family_label"] == f), None)
        ys = [num(r, k) for k in keys]
        ax.plot(thresholds, ys, marker="o", markersize=12, lw=3.0,
                color=FAM_COLOR[f], label=f)
        for i, (x, y) in enumerate(zip(thresholds, ys)):
            # last point: label to the right; others: label above
            if i == len(thresholds) - 1:
                dx, dy, ha, va = 12, 0, "left", "center"
            else:
                dx, dy, ha, va = 0, 14, "center", "bottom"
            ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=12,
                        color=FAM_COLOR[f], weight="bold", ha=ha, va=va)
    ax.set_title("OOD success vs precision threshold  ·  budget 200", fontsize=18, weight="bold", pad=14)
    ax.set_ylabel("OOD success (%)", fontsize=14)
    ax.set_xlabel("Precision threshold (nearest distance)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.25, len(thresholds) - 0.5)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    style_axes(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.015, "Loose 5cm thresholds inflate apparent success — 1cm is the honest precision metric.",
             ha="center", fontsize=11, color="#6b6b6b", style="italic")
    out = OUT_DIR / "threshold_sweep.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_family_precision_end():
    rows = [r for r in load("overall_by_family_split_budget.csv")
            if r["split"] == "ood" and int(r["budget"]) == 200]
    precisions = []
    ends = []
    for f in FAMS:
        r = next((r for r in rows if r["family_label"] == f), None)
        precisions.append(num(r, "success_at_1cm_pct"))
        ends.append(num(r, "end_distance_cm"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, H_IN), dpi=DPI, gridspec_kw={"wspace": 0.32})

    short_labels = ["Scratch CNN", "Frozen R-18", "Partial R-18"]
    colors = [FAM_COLOR[f] for f in FAMS]
    bars1 = ax1.bar(short_labels, precisions, color=colors, edgecolor="white", lw=1.5, width=0.6)
    for bar, val in zip(bars1, precisions):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=14, weight="bold", color="#222")
    ax1.set_title("Precision: success@1cm (higher better)", fontsize=15, weight="bold", pad=10)
    ax1.set_ylabel("OOD success@1cm (%)", fontsize=13)
    ax1.set_ylim(0, max(precisions) * 1.22)
    style_axes(ax1)
    ax1.tick_params(axis="x", labelsize=12, rotation=0)

    bars2 = ax2.bar(short_labels, ends, color=colors, edgecolor="white", lw=1.5, width=0.6)
    for bar, val in zip(bars2, ends):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f} cm",
                 ha="center", va="bottom", fontsize=14, weight="bold", color="#222")
    ax2.set_title("Stability: end distance (lower better)", fontsize=15, weight="bold", pad=10)
    ax2.set_ylabel("OOD end distance (cm)", fontsize=13)
    ax2.set_ylim(0, max(ends) * 1.22)
    style_axes(ax2)
    ax2.tick_params(axis="x", labelsize=12, rotation=0)

    fig.suptitle("Pretrained encoders trade precision for stability  ·  OOD, budget 200",
                 fontsize=17, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "family_precision_end.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_axis_threshold_grid():
    """2x2 panel: one panel per axis. Within each panel, 3 thresholds (1/2/5 cm)
    grouped by family. OOD only, budget 50 (3-seed)."""
    rows = [r for r in load("ood_axis_by_family_budget.csv") if int(r["budget"]) == 50]
    axes_order = ["Color", "Camera viewpoint", "Lighting", "Spatial distribution"]
    thresh_keys = ["success_at_1cm_pct", "success_at_2cm_pct", "success_at_5cm_pct"]
    thresh_labels = ["@1 cm", "@2 cm", "@5 cm"]

    fig, axarr = plt.subplots(2, 2, figsize=(W_IN, H_IN), dpi=DPI,
                              gridspec_kw={"hspace": 0.42, "wspace": 0.18})
    panels = axarr.flatten()

    for i, ax_name in enumerate(axes_order):
        ax = panels[i]
        x = np.arange(len(thresh_labels))
        width = 0.26
        for j, fam in enumerate(FAMS):
            r = next((r for r in rows if r["family_label"] == fam and r["axis_label"] == ax_name), None)
            ys = [num(r, k) if r else 0.0 for k in thresh_keys]
            offs = (j - 1) * width
            bars = ax.bar(x + offs, ys, width,
                          label=fam if i == 0 else None,
                          color=FAM_COLOR[fam], edgecolor="white", lw=1.0)
            for bar, val in zip(bars, ys):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.6,
                        f"{val:.0f}", ha="center", va="bottom",
                        fontsize=9, color="#222")
        title_color = "#b8332a" if ax_name == "Spatial distribution" else "#1a1a1a"
        ax.set_title(ax_name, fontsize=14, weight="bold", pad=8, color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels(thresh_labels, fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=10)
        if i % 2 == 0:
            ax.set_ylabel("OOD success (%)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#e6e3dc", lw=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("OOD success per visual axis × precision threshold  ·  budget 50, 3 seeds",
                 fontsize=17, weight="bold", y=0.99)
    fig.legend(loc="lower center", ncol=3, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = OUT_DIR / "axis_threshold_grid.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("rendering slide charts...")
    chart_axis_bars()
    chart_threshold_sweep()
    chart_family_precision_end()
    chart_axis_threshold_grid()
    print(f"done. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
