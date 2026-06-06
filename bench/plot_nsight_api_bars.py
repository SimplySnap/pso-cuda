#!/usr/bin/env python3
"""Shape A1 for Table A (CUDA API breakdown): stacked bars, log-y, per cell.

Single panel. Six bars at np=4, one per (D, N) cell. Each bar stacks the
five API components (cudaMemcpy, cudaDeviceSync, cudaLaunchKernel,
cudaMemcpyFromSymbol, other). The D=300 N=8M cell is annotated as OOM.

Reads bench/nsight_matrix.csv, writes bench/fig_nsight_api_bars.png.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

CSV = os.path.join(HERE, "nsight_matrix.csv")
OUT = os.path.join(HERE, "fig_nsight_api_bars.png")

# nsys raw-name substring → canonical short label
API_MAP = [
    ("cudaMemcpyFromSymbol", "cudaMemcpyFromSymbol"),
    ("cudaMemcpy",           "cudaMemcpy"),
    ("cudaDeviceSynchronize","cudaDeviceSync"),
    ("cudaLaunchKernel",     "cudaLaunchKernel"),
]
# Stack order bottom → top; choose so the biggest contributors land bottom
STACK_ORDER = [
    ("cudaDeviceSync",       "#8172B2"),  # purple — biggest at large N
    ("cudaMemcpy",           "#C44E52"),  # red — biggest at small N
    ("cudaLaunchKernel",     "#DD8452"),  # orange
    ("cudaMemcpyFromSymbol", "#888888"),  # gray (one-time startup)
    ("other",                "#C0C0C0"),  # light gray
]

NP_PICK = 4
CELLS = [
    (30,  1024),
    (30,  2097152),
    (30,  8388608),
    (300, 1024),
    (300, 2097152),
    (300, 8388608),  # OOM placeholder
]


def canonicalize(name):
    for needle, canon in API_MAP:
        if needle in name:
            return canon
    return "other"


def cell_sums(df, D, N, ranks):
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)
             & (df["category"] == "api")]
    if sub.empty:
        return None
    out = {comp: 0.0 for comp, _ in STACK_ORDER}
    for _, r in sub.iterrows():
        out[canonicalize(r["name"])] += r["total_ms"]
    return out


def fmt_N(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


df = pd.read_csv(CSV)

# --- generous figure size & padding ----------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.18)

x = list(range(len(CELLS)))
labels = [f"D={D}\nN={fmt_N(N)}" for D, N in CELLS]
sums = [cell_sums(df, D, N, NP_PICK) for D, N in CELLS]

bottoms = [0.0] * len(CELLS)
for comp, color in STACK_ORDER:
    vals = [(s[comp] if s else 0.0) for s in sums]
    ax.bar(x, vals, bottom=bottoms, color=color, label=comp,
           width=0.62, edgecolor="white", linewidth=0.7)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

# Total annotation above each bar
for i, s in enumerate(sums):
    if s is None:
        # OOM placeholder
        ax.bar([i], [0.4], color="none", edgecolor="#B22222",
               linestyle="--", linewidth=1.5, width=0.62)
        ax.text(i, 0.45, "OOM\n(24 GB cap)",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#B22222")
    else:
        total = sum(s[c] for c, _ in STACK_ORDER)
        ax.text(i, total * 1.40,
                f"{total/1000:.2f} s" if total >= 1000 else f"{total:.1f} ms",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")

# Visual separator between D=30 and D=300
ax.axvline(2.5, color="#888", linestyle=":", alpha=0.6, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=STYLE["tick_label_size"])
ax.set_ylabel("CUDA API time (ms, log scale)",
              fontsize=STYLE["axis_label_size"], labelpad=10)
ax.set_yscale("log")
ax.set_ylim(0.05, 5e4)
ax.tick_params(axis="y", labelsize=STYLE["tick_label_size"], pad=4)
ax.tick_params(axis="x", pad=8)

# D labels above each group (clear of the value annotations)
ax.text(1.0, 6e4, "D = 30",  ha="center", fontsize=12,
        fontweight="bold", color="#444")
ax.text(4.5, 6e4, "D = 300", ha="center", fontsize=12,
        fontweight="bold", color="#444")

ax.set_title(
    "Table A — CUDA API time at np=4 (Nsight, rank 0, ring)",
    fontsize=STYLE["panel_title_size"] + 2, pad=22)

ax.grid(alpha=STYLE["grid_alpha"], axis="y", which="both")
# Legend OUTSIDE on the right so it never overlaps bars
ax.legend(fontsize=STYLE["legend_size"],
          loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=False, title="API call")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"wrote {OUT}")
