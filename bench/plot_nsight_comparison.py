#!/usr/bin/env python3
"""Nsight CUDA-API and kernel breakdowns at two regimes — the inversion story.

Reads the canonical numbers from bench/nsys_summary.txt (small-N M4 baseline)
and bench/nsys_summary_largeN.txt. Rather than parsing those text files,
numbers are hardcoded below — they are stable artifacts of fixed slurm runs
and are already cited verbatim in §3.7 of bench/M4_REPORT.md.

Writes bench/fig_nsight_comparison.png as a 2-panel side-by-side figure.

Left panel: small-N regime (ring np=2, N=1024, D=30, iters=200).
            cudaMemcpy = 7.5× kernel time.
Right panel: large-N regime (ring np=4, N=2M, D=100, iters=100).
             Kernel time = 2.3× cudaMemcpy.
"""

import os
import sys
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

# ---- regime numbers (milliseconds) ----------------------------------
SMALL_N_API = [          # bench/nsys_summary.txt, rank 0
    ("cudaMemcpy",            28.35, "#C44E52"),
    ("cudaMemcpyFromSymbol",  11.07, "#7F3F4D"),  # one-time startup
    ("cudaLaunchKernel",       5.41, "#8172B2"),
    ("cudaEventRecord",        2.27, "#888888"),
]
SMALL_N_KERN = [
    ("kernel_eval_and_pbest",  2.15, "#4C72B0"),
    ("kernel_draw_rng",        0.67, "#55A868"),
    ("CUB DeviceReduce",       0.38, "#8172B2"),
    ("kernel_update",          0.31, "#C44E52"),
    ("kernel_commit_gbest",    0.26, "#888888"),
]

LARGE_N_API = [          # bench/nsys_summary_largeN.txt, rank 0
    ("cudaDeviceSynchronize", 3018.0, "#888888"),  # the pre-callback drains
    ("cudaMemcpy",            1310.0, "#C44E52"),
    ("cudaMemcpyFromSymbol",   154.0, "#7F3F4D"),  # one-time startup
    ("cudaLaunchKernel",         5.65, "#8172B2"),
]
LARGE_N_KERN = [
    ("kernel_eval_and_pbest", 1224.0, "#4C72B0"),
    ("kernel_update",         1200.0, "#C44E52"),
    ("kernel_draw_rng",        571.0, "#55A868"),
    ("kernel_curand_init",      16.0, "#888888"),  # one-time startup
]


def draw_stacked(ax, api_data, kern_data, title, headline):
    """Two stacked bars: 'CUDA API' on the left, 'Kernels' on the right.
    Each bar made of (label, ms, color) tuples stacked from bottom.
    Annotate each segment whose share is >5% of the bar's total."""
    bar_x = [0, 1]
    bar_labels = ["CUDA API", "Kernels"]
    data_sets = [api_data, kern_data]

    for x, label, data in zip(bar_x, bar_labels, data_sets):
        bottom = 0.0
        total = sum(v for _, v, _ in data)
        for name, v, color in data:
            ax.bar(x, v, bottom=bottom, color=color, width=0.6,
                   edgecolor="white", linewidth=0.5)
            if v / total > 0.05:
                ax.text(x, bottom + v / 2.0, f"{name}\n{v:.1f} ms",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if v / total > 0.15 else "black")
            bottom += v
        ax.text(x, bottom * 1.04, f"Σ = {total:.1f} ms",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, fontsize=STYLE["tick_label_size"])
    ax.set_ylabel("time (ms)", fontsize=STYLE["axis_label_size"])
    ax.set_title(f"{title}\n{headline}",
                 fontsize=STYLE["panel_title_size"])
    ax.tick_params(axis="y", labelsize=STYLE["tick_label_size"])
    ax.grid(alpha=STYLE["grid_alpha"], axis="y")
    ax.set_yscale("log")


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))

draw_stacked(
    ax_l, SMALL_N_API, SMALL_N_KERN,
    title="Small N — ring np=2, N=1024, D=30, 200 iters",
    headline=r"cudaMemcpy / kernel time = 28.4 / 3.8 = 7.5×")

draw_stacked(
    ax_r, LARGE_N_API, LARGE_N_KERN,
    title="Large N — ring np=4, N=2M, D=100, 100 iters",
    headline=r"kernel / cudaMemcpy time = 3013 / 1310 = 2.3×")

fig.suptitle("Nsight Systems regime inversion: host-staging vs GPU compute",
             y=1.02, fontsize=STYLE["panel_title_size"] + 1)
fig.tight_layout()
OUT = os.path.join(HERE, "fig_nsight_comparison.png")
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
