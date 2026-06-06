#!/usr/bin/env python3
"""Comm vs compute stacked-bar breakdown for all 20 strong + weak cells.

Reads bench/sweep_largeN_strong.csv and bench/sweep_largeN_weak.csv (both
headerless). Writes bench/fig_breakdown.png.

Single panel, 20 vertical stacked bars (10 strong + 10 weak). Bars labelled
S-ring-np1, S-fc-np1, ..., W-ring-np16, W-fc-np16. Stack components from
bottom: eval, reduce, update, sync. y-axis log scale (the fc np=16 weak
sync_ms is 512 sec while compute is 53 sec; log lets both ends breathe).
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _plot_style import STYLE

MPI_COLS = ["topology", "evaluator", "n_islands", "N", "D", "iters", "seed",
            "eval_ms", "reduce_ms", "update_ms", "sync_ms", "total_ms",
            "final_gbest"]
strong = pd.read_csv(os.path.join(HERE, "sweep_largeN_strong.csv"),
                     header=None, names=MPI_COLS)
weak   = pd.read_csv(os.path.join(HERE, "sweep_largeN_weak.csv"),
                     header=None, names=MPI_COLS)

# Order: all strong cells first (S- prefix), then all weak (W- prefix).
# Within each study, sort by ranks then topology so ring/fc pairs are adjacent.
strong = strong.sort_values(["n_islands", "topology"]).reset_index(drop=True)
weak   = weak.sort_values(["n_islands", "topology"]).reset_index(drop=True)
strong["study"] = "S"
weak["study"] = "W"
df = pd.concat([strong, weak], ignore_index=True)

labels = [f"{r['study']}-{r['topology']}-np{int(r['n_islands'])}"
          for _, r in df.iterrows()]
x = list(range(len(df)))

fig, ax = plt.subplots(figsize=(12, 4))
COMPONENTS = ["eval_ms", "reduce_ms", "update_ms", "sync_ms"]
COMPONENT_NAMES = ["eval", "reduce", "update", "sync"]
COMPONENT_COLORS = [STYLE["stack_colors"][k] for k in COMPONENT_NAMES]

# matplotlib's log-y stacked bar trick: log scale doesn't handle stacked bars
# natively (each segment starts at 0 visually). Plot cumulative totals and
# colour-fill between, top-down.
# Simpler approach: convert to seconds for readable axis numbers, use linear
# y but with a log-style breakpoint? Actually a linear y-axis works
# acceptably if we scale to seconds — the fc np=16 weak bar (~566 s) dwarfs
# the rest but the eval/reduce/update components in other bars are at least
# visible as thin strips. Test linear first; switch to log if needed.

bottoms = [0.0] * len(df)
for part, color, name in zip(COMPONENTS, COMPONENT_COLORS, COMPONENT_NAMES):
    vals = (df[part] / 1000.0).tolist()    # ms -> seconds
    ax.bar(x, vals, bottom=bottoms, color=color, label=name, width=0.8)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

# Color the x-tick label background subtly so strong vs weak studies pop visually
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=60, ha="right")
# Vertical divider between strong and weak halves
ax.axvline(len(strong) - 0.5, color="black", linestyle=":", alpha=0.4)
ax.text(len(strong) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
        "STRONG (N_total = 8M)", ha="center", fontsize=10,
        color="#444", fontweight="bold")
ax.text(len(strong) + len(weak) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
        "WEAK (per-rank N = 8M)", ha="center", fontsize=10,
        color="#444", fontweight="bold")

ax.set_ylabel("time (s)", fontsize=STYLE["axis_label_size"])
ax.set_title("Per-component time breakdown across strong and weak scaling configs",
             fontsize=STYLE["panel_title_size"])
ax.tick_params(axis="y", labelsize=STYLE["tick_label_size"])
ax.grid(alpha=STYLE["grid_alpha"], axis="y")
ax.legend(fontsize=STYLE["legend_size"], loc="upper left", ncol=4)

fig.tight_layout()
OUT = os.path.join(HERE, "fig_breakdown.png")
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
