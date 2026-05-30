#!/usr/bin/env python3
"""Strong scaling speedup + efficiency at N_total = 8M, D = 100.

Reads bench/sweep_largeN_strong.csv (headerless MPI --csv_path output) and
bench/sweep_largeN_strong_baseline.csv (with header). Writes
bench/fig_strong_scaling.png as a 2-panel side-by-side figure.

Left panel: speedup vs ranks, four lines (ring vs MPI np=1, ring vs pso_cuda,
fc vs MPI np=1, fc vs pso_cuda). Diagonal y=p reference.
Right panel: efficiency vs ranks, two lines (ring, fc), 1.0 reference.
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

mpi = pd.read_csv(os.path.join(HERE, "sweep_largeN_strong.csv"),
                  header=None, names=MPI_COLS)
base = pd.read_csv(os.path.join(HERE, "sweep_largeN_strong_baseline.csv"))

t_single = float(base[base["N"] == 8388608]["total_ms"].iloc[0])

# Compute speedup and efficiency per topology against MPI np=1 baseline
rows = []
for topo, g in mpi.sort_values("n_islands").groupby("topology"):
    t1 = float(g[g["n_islands"] == 1]["total_ms"].iloc[0])
    for _, r in g.iterrows():
        rows.append({
            "topology": topo, "ranks": int(r["n_islands"]),
            "speedup_vs_mpi1":     t1       / r["total_ms"],
            "speedup_vs_single":   t_single / r["total_ms"],
            "efficiency":          (t1 / r["total_ms"]) / r["n_islands"],
        })
df = pd.DataFrame(rows)

fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(11, 4))

# -------- LEFT: speedup --------
for topo, color, ls, marker in [
    ("ring", STYLE["ring_color"], STYLE["ring_ls"], STYLE["ring_marker"]),
    ("fc",   STYLE["fc_color"],   STYLE["fc_ls"],   STYLE["fc_marker"]),
]:
    g = df[df["topology"] == topo].sort_values("ranks")
    ax_s.plot(g["ranks"], g["speedup_vs_mpi1"],
              color=color, linestyle=ls, marker=marker, markersize=7,
              label=f"{topo} vs MPI np=1")
    ax_s.plot(g["ranks"], g["speedup_vs_single"],
              color=color, linestyle=STYLE["baseline_ls"],
              marker=marker, markersize=6, markerfacecolor="white",
              label=f"{topo} vs pso_cuda")

# Annotate ring np=16
r16 = df[(df["topology"] == "ring") & (df["ranks"] == 16)].iloc[0]
ax_s.annotate(f"{r16['speedup_vs_mpi1']:.2f}×",
              xy=(16, r16["speedup_vs_mpi1"]),
              xytext=(8, r16["speedup_vs_mpi1"] + 1.5),
              fontsize=9, color=STYLE["ring_color"],
              arrowprops=dict(arrowstyle="->", color=STYLE["ring_color"]))

# ideal y = p line
xs = sorted(df["ranks"].unique())
ax_s.plot(xs, xs, color="gray", linestyle=":", alpha=0.7, label="ideal y=p")

ax_s.set_xscale("log", base=2)
ax_s.set_xticks([1, 2, 4, 8, 16])
ax_s.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax_s.set_xlabel("ranks (np)", fontsize=STYLE["axis_label_size"])
ax_s.set_ylabel("speedup", fontsize=STYLE["axis_label_size"])
ax_s.set_title("Speedup", fontsize=STYLE["panel_title_size"])
ax_s.tick_params(labelsize=STYLE["tick_label_size"])
ax_s.grid(alpha=STYLE["grid_alpha"])
ax_s.legend(fontsize=STYLE["legend_size"], loc="upper left")

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
              xytext=(9, r16["efficiency"] - 0.15),
              fontsize=9, color=STYLE["ring_color"],
              arrowprops=dict(arrowstyle="->", color=STYLE["ring_color"]))

ax_e.set_xscale("log", base=2)
ax_e.set_xticks([1, 2, 4, 8, 16])
ax_e.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax_e.set_ylim(0, 1.1)
ax_e.set_xlabel("ranks (np)", fontsize=STYLE["axis_label_size"])
ax_e.set_ylabel("efficiency (speedup / p, vs MPI np=1)",
                fontsize=STYLE["axis_label_size"])
ax_e.set_title("Parallel efficiency", fontsize=STYLE["panel_title_size"])
ax_e.tick_params(labelsize=STYLE["tick_label_size"])
ax_e.grid(alpha=STYLE["grid_alpha"])
ax_e.legend(fontsize=STYLE["legend_size"], loc="lower left")

fig.suptitle("Strong scaling at N_total = 8M, D = 100",
             y=1.02, fontsize=STYLE["panel_title_size"] + 1)
fig.tight_layout()
OUT = os.path.join(HERE, "fig_strong_scaling.png")
fig.savefig(OUT, dpi=STYLE["dpi"], bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
