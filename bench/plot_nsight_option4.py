#!/usr/bin/env python3
"""Option 4 from the brainstorm: regime-overview stacked bars at np=4.

One stacked bar per (D, N) cell at fixed np=4. Each bar shows the major
time consumers stacked together: GPU kernel components on one side and
CUDA API components on the other. Six bars total (D in {30, 300} × N in
{1024, 2M, 8M}); the D=300 N=8M cell is annotated OOM.

Reads bench/nsight_matrix.csv (long format: D, N, ranks, category, name,
total_ms, percent). Writes bench/fig_nsight_option4.png.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

CSV = os.path.join(HERE, "nsight_matrix.csv")
OUT = os.path.join(HERE, "fig_nsight_option4.png")

# Map nsys raw names to canonical labels used as stack components.
API_MAP = [
    ("cudaMemcpyFromSymbol", "cudaMemcpyFromSymbol"),
    ("cudaMemcpy",           "cudaMemcpy"),
    ("cudaDeviceSynchronize","cudaDeviceSync"),
    ("cudaLaunchKernel",     "cudaLaunchKernel"),
]
KERNEL_MAP = [
    ("kernel_eval_and_pbest", "eval_and_pbest"),
    ("kernel_update",         "update"),
    ("kernel_draw_rng",       "draw_rng"),
    ("DeviceReduce",          "CUB_ArgMin"),
    ("kernel_commit_gbest",   "commit_gbest"),
]

# Stack order (bottom → top) and colors. Kernels on the bottom (compute),
# API on the top (host-side overhead).
STACK_ORDER = [
    ("eval_and_pbest",       "#4C72B0"),   # blue
    ("update",               "#55A868"),   # green
    ("draw_rng",             "#DD8452"),   # orange
    ("kernel_other",         "#A8A8A8"),   # light gray
    ("cudaMemcpy",           "#C44E52"),   # red
    ("cudaDeviceSync",       "#8172B2"),   # purple
    ("api_other",            "#555555"),   # dark gray
]

def canonicalize(name, kind):
    table = API_MAP if kind == "api" else KERNEL_MAP
    for needle, canon in table:
        if needle in name:
            return canon
    return f"{kind}_other"

# Pick np=4 (middle of the rank range; we know kernel times are np-invariant).
NP_PICK = 4

# Cells in the order we want to display (rows are read left→right).
CELLS = [
    (30,  1024),
    (30,  2097152),
    (30,  8388608),
    (300, 1024),
    (300, 2097152),
    (300, 8388608),   # OOM; rendered as an annotated empty slot
]

def cell_sums(df, D, N, ranks):
    """Return dict of canonical -> total_ms for one (D, N, ranks) cell."""
    sub = df[(df["D"] == D) & (df["N"] == N) & (df["ranks"] == ranks)]
    if sub.empty:
        return None
    out = {c: 0.0 for c, _ in STACK_ORDER}
    for _, r in sub.iterrows():
        canon = canonicalize(r["name"], r["category"])
        out[canon] = out.get(canon, 0.0) + r["total_ms"]
    return out


def fmt_N(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


df = pd.read_csv(CSV)

# Build the bars.
labels, sums = [], []
for D, N in CELLS:
    s = cell_sums(df, D, N, NP_PICK)
    labels.append(f"D={D}\nN={fmt_N(N)}")
    sums.append(s)

fig, ax = plt.subplots(figsize=(11, 5))
x = list(range(len(CELLS)))

bottoms = [0.0] * len(CELLS)
for comp, color in STACK_ORDER:
    vals = [(s[comp] if s else 0.0) for s in sums]
    ax.bar(x, vals, bottom=bottoms, color=color, label=comp,
           width=0.65, edgecolor="white", linewidth=0.6)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

# Annotate total ms on top of each bar
for i, (s, lab) in enumerate(zip(sums, labels)):
    if s is None:
        ax.text(i, 1.0, "OOM\n(24 GB cap)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#B22222")
        # Faint dashed rectangle as a "placeholder" for the missing cell
        ax.bar([i], [0.5], color="none", edgecolor="#B22222",
               linestyle="--", linewidth=1.2, width=0.65)
    else:
        total = sum(s[c] for c, _ in STACK_ORDER)
        ax.text(i, total * 1.08,
                f"{total/1000:.2f} s" if total >= 1000 else f"{total:.0f} ms",
                ha="center", fontsize=9, fontweight="bold", color="#333")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=STYLE["tick_label_size"])
ax.set_ylabel("time (ms, log scale)", fontsize=STYLE["axis_label_size"])
ax.set_yscale("log")
ax.tick_params(axis="y", labelsize=STYLE["tick_label_size"])
ax.set_title(
    f"Per-component time breakdown at np={NP_PICK} (Nsight, rank 0, ring)",
    fontsize=STYLE["panel_title_size"] + 1)

# Visual separator between D=30 and D=300 columns
ax.axvline(2.5, color="#666", linestyle=":", alpha=0.5)
ax.text(1.0, ax.get_ylim()[1] * 1.2, "D = 30",
        ha="center", fontsize=11, fontweight="bold", color="#444")
ax.text(4.5, ax.get_ylim()[1] * 1.2, "D = 300",
        ha="center", fontsize=11, fontweight="bold", color="#444")

ax.grid(alpha=STYLE["grid_alpha"], axis="y", which="both")
ax.legend(fontsize=STYLE["legend_size"], loc="upper left",
          ncol=2, framealpha=0.95)

# Clean spines
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
