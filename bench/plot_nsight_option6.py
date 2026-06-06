#!/usr/bin/env python3
"""Option 6 from the brainstorm: kernel/cudaMemcpy ratio vs N.

Single-panel line chart showing the host-staging vs compute inversion.
- x-axis: per-rank N (log scale)
- y-axis: ratio = total GPU kernel time / total cudaMemcpy time (log scale)
- One line per (D, ranks) pair. At fixed D the ratio shape is essentially
  identical across ranks (kernel times are np-invariant per §3.7 finding 3
  and cudaMemcpy scales with m × D not p), so showing all 5 np values
  produces 10 nearly-overlapping lines. We show 3 representative np values
  per D instead: {1, 4, 16}.
- y = 1 reference line marks "kernels = cudaMemcpy" — crossing it from
  below to above is the regime inversion.

Reads bench/nsight_matrix.csv. Writes bench/fig_nsight_option6.png.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

CSV = os.path.join(HERE, "nsight_matrix.csv")
OUT = os.path.join(HERE, "fig_nsight_option6.png")


def total_for_category_and_name(df, D, N, ranks, category, needle):
    """Sum total_ms for rows in (D, N, ranks) whose category matches and
    whose name contains the needle substring."""
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)
             & (df["category"] == category)
             & (df["name"].str.contains(needle, regex=False))]
    return float(sub["total_ms"].sum())


def kernel_total(df, D, N, ranks):
    """Sum of ALL GPU kernel time at this cell."""
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)
             & (df["category"] == "kernel")]
    return float(sub["total_ms"].sum())


def cudamemcpy_total(df, D, N, ranks):
    """cudaMemcpy time (excludes cudaMemcpyFromSymbol startup)."""
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)
             & (df["category"] == "api")
             & (df["name"] == "cudaMemcpy")]
    return float(sub["total_ms"].sum())


df = pd.read_csv(CSV)

DS = [30, 300]
NS = [1024, 2097152, 8388608]
NP_SHOWN = [1, 4, 16]

D_COLORS = {30: STYLE["ring_color"], 300: STYLE["fc_color"]}
NP_STYLES = {1: ":", 4: "-", 16: "--"}
NP_MARKERS = {1: "v", 4: "o", 16: "^"}

fig, ax = plt.subplots(figsize=(7.5, 4.5))

for D in DS:
    for ranks in NP_SHOWN:
        xs, ys = [], []
        for N in NS:
            k = kernel_total(df, D, N, ranks)
            m = cudamemcpy_total(df, D, N, ranks)
            if k > 0 and m > 0:
                xs.append(N)
                ys.append(k / m)
        if xs:
            label = f"D={D}, np={ranks}"
            ax.plot(xs, ys,
                    color=D_COLORS[D],
                    linestyle=NP_STYLES[ranks],
                    marker=NP_MARKERS[ranks], markersize=7,
                    linewidth=1.8,
                    label=label)

# Inversion threshold
ax.axhline(1.0, color="#333", linestyle="-", alpha=0.7, linewidth=1.2)
ax.text(NS[0] * 1.1, 1.05, "kernel = cudaMemcpy (inversion)",
        fontsize=9, color="#333")

# Shaded regions to communicate the two regimes
xlim = (NS[0] * 0.7, NS[-1] * 1.6)
ax.fill_between(xlim, 0.01, 1.0, color="#C44E52", alpha=0.06,
                label=None)
ax.fill_between(xlim, 1.0, 100.0, color="#4C72B0", alpha=0.06,
                label=None)
ax.text(NS[-1] * 1.15, 0.15, "cudaMemcpy\ndominates",
        fontsize=9, color="#C44E52", ha="center")
ax.text(NS[-1] * 1.15, 3.0, "kernels\ndominate",
        fontsize=9, color="#4C72B0", ha="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*xlim)
ax.set_ylim(0.03, 8.0)
ax.set_xlabel("per-rank N", fontsize=STYLE["axis_label_size"])
ax.set_ylabel("kernel time / cudaMemcpy time",
              fontsize=STYLE["axis_label_size"])
ax.set_title(
    "Compute–vs–host-staging inversion as N grows (Nsight, rank 0, ring)",
    fontsize=STYLE["panel_title_size"] + 1)
ax.tick_params(labelsize=STYLE["tick_label_size"])

# Custom x-tick labels for the three N values
ax.set_xticks(NS)
ax.set_xticklabels([f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"
                    for n in NS])

ax.grid(alpha=STYLE["grid_alpha"], which="both")
ax.legend(fontsize=STYLE["legend_size"], loc="upper left",
          ncol=2, framealpha=0.95)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
