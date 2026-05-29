#!/usr/bin/env python3
"""Convergence vs ranks per (D, topology). Reads bench/correctness_largeN.csv,
writes bench/fig_correctness.png.

Two panels: D=100 left, D=300 right. Each panel shows ring vs fc vs pso_cuda
baseline. The fc np=1 D=300 anomaly (gbest=3,359, off-scale) gets an
annotated arrow rather than warping the y-axis.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

CSV = os.path.join(HERE, "correctness_largeN.csv")
OUT = os.path.join(HERE, "fig_correctness.png")

df = pd.read_csv(CSV)
df = df[df["evaluator"] == "rastrigin"]

# (D, N) cells in the order we want them displayed
CELLS = [(100, 524288), (300, 131072)]

# Per-cell y-axis crop so the fc np=1 D=300 outlier (3,359) doesn't squash
# the rest of the data
Y_CROP = {100: 250.0, 300: 1800.0}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, (D, N) in zip(axes, CELLS):
    sub = df[(df["D"] == D) & (df["N"] == N)]
    single = sub[sub["impl"] == "single"]["final_gbest"].iloc[0]

    for topo, color, ls, marker in [
        ("ring", STYLE["ring_color"], STYLE["ring_ls"], STYLE["ring_marker"]),
        ("fc",   STYLE["fc_color"],   STYLE["fc_ls"],   STYLE["fc_marker"]),
    ]:
        m = sub[(sub["impl"] == "mpi") & (sub["topology"] == topo)] \
                .sort_values("n_ranks")
        ax.plot(m["n_ranks"], m["final_gbest"],
                color=color, linestyle=ls, marker=marker, markersize=7,
                label=topo)

    # pso_cuda single-GPU baseline as horizontal reference
    ax.axhline(single,
               color=STYLE["single_color"], linestyle=STYLE["baseline_ls"],
               label=f"pso_cuda = {single:.1f}")

    # crop y so the fc np=1 D=300 anomaly doesn't dominate
    crop = Y_CROP[D]
    ax.set_ylim(0, crop)

    # annotate any data point that exceeded the crop
    for _, row in sub[sub["impl"] == "mpi"].iterrows():
        if row["final_gbest"] > crop:
            ax.annotate(
                f"{row['final_gbest']:,.0f}",
                xy=(row["n_ranks"], crop * 0.97),
                xytext=(row["n_ranks"], crop * 0.78),
                ha="center", fontsize=9, color=STYLE["fc_color"],
                arrowprops=dict(arrowstyle="->", color=STYLE["fc_color"]))

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("ranks (np)", fontsize=STYLE["axis_label_size"])
    ax.set_ylabel("final gbest (Rastrigin, lower = better)",
                  fontsize=STYLE["axis_label_size"])
    ax.set_title(f"D = {D}, per-rank N = {N:,}",
                 fontsize=STYLE["panel_title_size"])
    ax.tick_params(labelsize=STYLE["tick_label_size"])
    ax.grid(alpha=STYLE["grid_alpha"])
    ax.legend(fontsize=STYLE["legend_size"], loc="best")

fig.tight_layout()
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
