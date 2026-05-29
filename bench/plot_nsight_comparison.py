#!/usr/bin/env python3
"""Nsight regime inversion: host-staging dominates at small N; GPU compute
dominates at large N.

Reads canonical numbers from bench/nsys_summary.txt (small-N M4 baseline)
and bench/nsys_summary_largeN.txt (Phase H large N). The numbers are
hardcoded here because they are stable artifacts of fixed slurm runs and
already cited in §3.7 of bench/M4_REPORT.md.

Writes bench/fig_nsight_comparison.png as a 2-panel figure. Each panel
shows the same two horizontal bars: total cudaMemcpy time, total GPU
kernel time. Linear x-axis per panel (so the within-regime ratio is
visible at a glance). The headline ratio is the dominant annotation.

The per-component breakdown (kernel_eval_and_pbest, kernel_update, etc.)
lives in the §3.7 table in M4_REPORT.md; this chart's job is to show
the inversion.
"""

import os
import sys
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

# ---- totals at each regime (milliseconds) -------------------------
# Small-N M4 baseline (ring np=2, N=1024, D=30, 200 iters, rank 0)
SMALL_N_CUDAMEMCPY_MS = 28.35       # sum of cudaMemcpy column from nsys
SMALL_N_KERNEL_MS     = 3.77        # sum of GPU kernel time
SMALL_N_LABEL = "ring np=2, N=1024, D=30, 200 iters"

# Large-N (ring np=4, N=2M, D=100, 100 iters, rank 0)
LARGE_N_CUDAMEMCPY_MS = 1310.0
LARGE_N_KERNEL_MS     = 3013.0
LARGE_N_LABEL = "ring np=4, N=2M, D=100, 100 iters"

C_CUDA   = STYLE["fc_color"]      # red — host-staging cost
C_KERNEL = STYLE["ring_color"]    # blue — productive GPU compute


def draw_panel(ax, label_top, label_bottom, val_top, val_bottom,
               headline, regime_title, units="ms"):
    """One panel: two horizontal bars (top = cudaMemcpy, bottom = kernel)
    with values annotated at the right end of each bar and the headline
    ratio drawn as the dominant text inside the panel."""
    y = [1, 0]
    vals = [val_top, val_bottom]
    colors = [C_CUDA, C_KERNEL]
    labels = [label_top, label_bottom]

    ax.barh(y, vals, color=colors, height=0.55, edgecolor="white", linewidth=1)

    # Numeric annotation at the right end of each bar
    x_max = max(vals)
    for yi, v, color in zip(y, vals, colors):
        ax.text(v + x_max * 0.02, yi,
                f"{v:,.1f} {units}",
                va="center", ha="left",
                fontsize=11, fontweight="bold", color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, x_max * 1.35)        # headroom for the right-end label
    ax.set_xlabel(f"time ({units})", fontsize=STYLE["axis_label_size"])
    ax.tick_params(axis="x", labelsize=STYLE["tick_label_size"])
    ax.grid(alpha=STYLE["grid_alpha"], axis="x")

    # Headline ratio centered in the panel — the visual hook
    ax.text(x_max * 0.6, 0.5, headline,
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#F5F5F5", edgecolor="#999",
                      linewidth=1.5))

    ax.set_title(regime_title, fontsize=STYLE["panel_title_size"] + 1,
                 pad=12)

    # Hide top + right spines for cleaner look
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4))

ratio_small = SMALL_N_CUDAMEMCPY_MS / SMALL_N_KERNEL_MS
draw_panel(
    ax_l,
    label_top="cudaMemcpy",
    label_bottom="GPU kernels",
    val_top=SMALL_N_CUDAMEMCPY_MS,
    val_bottom=SMALL_N_KERNEL_MS,
    headline=f"cudaMemcpy\nis {ratio_small:.1f}× kernels",
    regime_title=f"Small-N regime — {SMALL_N_LABEL}",
)

ratio_large = LARGE_N_KERNEL_MS / LARGE_N_CUDAMEMCPY_MS
draw_panel(
    ax_r,
    label_top="cudaMemcpy",
    label_bottom="GPU kernels",
    val_top=LARGE_N_CUDAMEMCPY_MS,
    val_bottom=LARGE_N_KERNEL_MS,
    headline=f"kernels\nare {ratio_large:.1f}× cudaMemcpy",
    regime_title=f"Large-N regime — {LARGE_N_LABEL}",
)

fig.suptitle(
    "Nsight Systems: the host-staging bottleneck inverts as N grows",
    y=1.04, fontsize=STYLE["panel_title_size"] + 2, fontweight="bold")
fig.tight_layout()
OUT = os.path.join(HERE, "fig_nsight_comparison.png")
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
