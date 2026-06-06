#!/usr/bin/env python3
"""Weak scaling total_ms + efficiency at per-rank N = 8M, D = 100.

Reads bench/sweep_largeN_weak.csv (headerless), writes
bench/fig_weak_scaling.png as a 2-panel side-by-side figure.

Left panel: total_ms vs ranks, log-y so ring's near-flat curve and the
fc np=16 explosion (565 sec vs ring's 101 sec) are both readable.
Right panel: weak-scaling efficiency = total_ms(np=1) / total_ms(np=p),
linear y, 1.0 reference.
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
mpi = pd.read_csv(os.path.join(HERE, "sweep_largeN_weak.csv"),
                  header=None, names=MPI_COLS)

# Compute efficiency = T(np=1) / T(np=p) per topology
rows = []
for topo, g in mpi.sort_values("n_islands").groupby("topology"):
    t1 = float(g[g["n_islands"] == 1]["total_ms"].iloc[0])
    for _, r in g.iterrows():
        rows.append({
            "topology": topo, "ranks": int(r["n_islands"]),
            "total_ms":   r["total_ms"],
            "efficiency": t1 / r["total_ms"],
        })
df = pd.DataFrame(rows)

fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11, 4))

# -------- LEFT: total_ms (log-y) --------
for topo, color, ls, marker in [
    ("ring", STYLE["ring_color"], STYLE["ring_ls"], STYLE["ring_marker"]),
    ("fc",   STYLE["fc_color"],   STYLE["fc_ls"],   STYLE["fc_marker"]),
]:
    g = df[df["topology"] == topo].sort_values("ranks")
    ax_t.plot(g["ranks"], g["total_ms"] / 1000.0,   # ms -> sec
              color=color, linestyle=ls, marker=marker, markersize=7,
              label=topo)

# Annotate the fc np=16 explosion
fc16 = df[(df["topology"] == "fc") & (df["ranks"] == 16)].iloc[0]
ax_t.annotate(f"{fc16['total_ms']/1000:.0f} s",
              xy=(16, fc16["total_ms"] / 1000),
              xytext=(6, fc16["total_ms"] / 1000 * 0.55),
              fontsize=9, color=STYLE["fc_color"],
              arrowprops=dict(arrowstyle="->", color=STYLE["fc_color"]))

ax_t.set_xscale("log", base=2)
ax_t.set_xticks([1, 2, 4, 8, 16])
ax_t.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax_t.set_yscale("log")
ax_t.set_xlabel("ranks (np)", fontsize=STYLE["axis_label_size"])
ax_t.set_ylabel("total_ms (seconds, log scale)",
                fontsize=STYLE["axis_label_size"])
ax_t.set_title("Total runtime", fontsize=STYLE["panel_title_size"])
ax_t.tick_params(labelsize=STYLE["tick_label_size"])
ax_t.grid(alpha=STYLE["grid_alpha"], which="both")
ax_t.legend(fontsize=STYLE["legend_size"], loc="upper left")

# -------- RIGHT: efficiency --------
for topo, color, ls, marker in [
    ("ring", STYLE["ring_color"], STYLE["ring_ls"], STYLE["ring_marker"]),
    ("fc",   STYLE["fc_color"],   STYLE["fc_ls"],   STYLE["fc_marker"]),
]:
    g = df[df["topology"] == topo].sort_values("ranks")
    ax_e.plot(g["ranks"], g["efficiency"],
              color=color, linestyle=ls, marker=marker, markersize=7,
              label=topo)

ax_e.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="ideal = 1.0")

r16 = df[(df["topology"] == "ring") & (df["ranks"] == 16)].iloc[0]
ax_e.annotate(f"{r16['efficiency']:.2f}",
              xy=(16, r16["efficiency"]),
              xytext=(9, r16["efficiency"] + 0.1),
              fontsize=9, color=STYLE["ring_color"],
              arrowprops=dict(arrowstyle="->", color=STYLE["ring_color"]))

ax_e.set_xscale("log", base=2)
ax_e.set_xticks([1, 2, 4, 8, 16])
ax_e.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax_e.set_ylim(0, 1.15)
ax_e.set_xlabel("ranks (np)", fontsize=STYLE["axis_label_size"])
ax_e.set_ylabel("efficiency = T(np=1) / T(np=p)",
                fontsize=STYLE["axis_label_size"])
ax_e.set_title("Weak-scaling efficiency",
               fontsize=STYLE["panel_title_size"])
ax_e.tick_params(labelsize=STYLE["tick_label_size"])
ax_e.grid(alpha=STYLE["grid_alpha"])
ax_e.legend(fontsize=STYLE["legend_size"], loc="lower left")

fig.suptitle("Weak scaling at per-rank N = 8M, D = 100",
             y=1.02, fontsize=STYLE["panel_title_size"] + 1)
fig.tight_layout()
OUT = os.path.join(HERE, "fig_weak_scaling.png")
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
