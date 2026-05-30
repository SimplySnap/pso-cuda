#!/usr/bin/env python3
"""Shape C2 for Table B (GPU kernel breakdown): per-component lines vs N.

Two panels side-by-side (D=30 left, D=300 right). Inside each panel: one
line per kernel component, ms (log-y) vs per-rank N (log-x). Fixed at np=4.

Reads bench/nsight_matrix.csv, writes bench/fig_nsight_kernel_lines.png.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

CSV = os.path.join(HERE, "nsight_matrix.csv")
OUT = os.path.join(HERE, "fig_nsight_kernel_lines.png")

KERNEL_MAP = [
    ("kernel_eval_and_pbest", "eval_and_pbest"),
    ("kernel_update",         "update"),
    ("kernel_draw_rng",       "draw_rng"),
    ("DeviceReduce",          "CUB_ArgMin"),
    ("kernel_commit_gbest",   "commit_gbest"),
]
COMPONENTS = [
    # name, color, marker
    ("eval_and_pbest", "#4C72B0", "o"),
    ("update",         "#55A868", "s"),
    ("draw_rng",       "#DBC75A", "^"),
    ("CUB_ArgMin",     "#3D8AA8", "D"),
    ("commit_gbest",   "#BFA94E", "v"),
]

NP_PICK = 4
DS = [30, 300]
NS = [1024, 2097152, 8388608]


def canonicalize(name):
    for needle, canon in KERNEL_MAP:
        if needle in name:
            return canon
    return None


def get_value(df, D, N, ranks, canon):
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)
             & (df["category"] == "kernel")]
    total = 0.0
    for _, r in sub.iterrows():
        if canonicalize(r["name"]) == canon:
            total += r["total_ms"]
    return total


def fmt_N(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


df = pd.read_csv(CSV)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
fig.subplots_adjust(left=0.08, right=0.83, top=0.85, bottom=0.16,
                    wspace=0.20)

for ax, D in zip(axes, DS):
    for comp, color, marker in COMPONENTS:
        xs, ys = [], []
        for N in NS:
            val = get_value(df, D, N, NP_PICK, comp)
            if val > 0:
                xs.append(N)
                ys.append(val)
        if xs:
            ax.plot(xs, ys, color=color, marker=marker, markersize=9,
                    linewidth=2.0, label=comp)

    if D == 300:
        ax.axvspan(NS[-1] * 0.65, NS[-1] * 1.6,
                   color="#B22222", alpha=0.08)
        ax.text(NS[-1], 0.15, "N=8M\nOOM",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#B22222")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([fmt_N(n) for n in NS])
    ax.set_xlim(NS[0] * 0.6, NS[-1] * 1.8)
    ax.set_ylim(0.05, 2e4)
    ax.set_xlabel("per-rank N", fontsize=STYLE["axis_label_size"], labelpad=8)
    ax.set_title(f"D = {D}",
                 fontsize=STYLE["panel_title_size"] + 1, pad=14)
    ax.tick_params(labelsize=STYLE["tick_label_size"], pad=4)
    ax.grid(alpha=STYLE["grid_alpha"], which="both")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("GPU kernel time (ms, log scale)",
                   fontsize=STYLE["axis_label_size"], labelpad=10)

axes[1].legend(fontsize=STYLE["legend_size"] + 1,
               loc="center left", bbox_to_anchor=(1.04, 0.5),
               frameon=False, title="GPU kernel", title_fontsize=10)

fig.suptitle(
    "Table B — GPU kernel time vs N at np=4 (Nsight, rank 0, ring)",
    y=0.97, fontsize=STYLE["panel_title_size"] + 2)

fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"wrote {OUT}")
