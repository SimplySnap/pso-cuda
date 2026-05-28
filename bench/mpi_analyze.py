#!/usr/bin/env python3
"""
mpi_analyze.py — figures and table for the M4 MPI scaling study.

Inputs (all in bench/):
  scaling_strong.csv     # ring + fc, total N=4096 split across ranks
  scaling_weak.csv       # ring + fc, per-rank N=1024
  scaling_baseline.csv   # pso_cuda at N=4096 and N=1024

Outputs (all in bench/):
  fig_mpi_scaling.png    # speedup + efficiency vs ranks, both baselines
  fig_mpi_breakdown.png  # stacked bar of eval/reduce/update/sync ms
  table_mpi.md           # markdown summary tables
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    return pd.read_csv(path)


def merge_speedup(scaling, baseline_n):
    """Compute speedup against both MPI np=1 and single-GPU pso_cuda."""
    out = scaling.copy().sort_values("n_islands").reset_index(drop=True)
    out["speedup_vs_mpi1"] = 0.0
    out["speedup_vs_single"] = 0.0
    out["efficiency_vs_mpi1"] = 0.0
    for topo, g in out.groupby("topology"):
        t_mpi1 = g.loc[g["n_islands"] == 1, "total_ms"].iloc[0]
        for idx in g.index:
            p = out.loc[idx, "n_islands"]
            t_p = out.loc[idx, "total_ms"]
            out.loc[idx, "speedup_vs_mpi1"]    = t_mpi1 / t_p
            out.loc[idx, "efficiency_vs_mpi1"] = (t_mpi1 / t_p) / p
            out.loc[idx, "speedup_vs_single"]  = baseline_n / t_p
    return out


def fig_scaling(strong, weak, base):
    """Two-panel: speedup and efficiency vs rank count, strong scaling."""
    t_single = float(base.loc[base["N"] == 4096, "total_ms"].iloc[0])
    s = merge_speedup(strong, t_single)

    fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.5))

    for topo, g in s.groupby("topology"):
        ax_sp.plot(g["n_islands"], g["speedup_vs_mpi1"],
                   marker="o", label=f"{topo} vs pso_{topo} -np 1")
        ax_sp.plot(g["n_islands"], g["speedup_vs_single"],
                   marker="s", linestyle="--",
                   label=f"{topo} vs pso_cuda (single GPU)")
        ax_eff.plot(g["n_islands"], g["efficiency_vs_mpi1"],
                    marker="o", label=f"{topo}")

    xs = sorted(s["n_islands"].unique())
    ax_sp.plot(xs, xs, color="gray", linestyle=":", label="ideal y = p")
    ax_sp.set_xlabel("number of ranks (p)")
    ax_sp.set_ylabel("speedup")
    ax_sp.set_title("strong scaling speedup (total N = 4096)")
    ax_sp.legend(fontsize=8)
    ax_sp.grid(alpha=0.3)

    ax_eff.axhline(1.0, color="gray", linestyle=":", label="ideal")
    ax_eff.set_xlabel("number of ranks (p)")
    ax_eff.set_ylabel("efficiency = speedup / p")
    ax_eff.set_title("strong scaling efficiency (vs np=1 baseline)")
    ax_eff.legend(fontsize=8)
    ax_eff.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_mpi_scaling.png"), dpi=120)
    plt.close(fig)


def fig_breakdown(strong, weak):
    """Stacked bar of eval/reduce/update/sync ms per (topology, ranks).
    Two panels: strong (total N=4096) and weak (per-rank N=1024)."""
    parts = ["eval_ms", "reduce_ms", "update_ms", "sync_ms"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, (df, title) in zip(axes, [
        (strong, "strong scaling: total N = 4096"),
        (weak,   "weak scaling: N = 1024 per rank"),
    ]):
        df = df.sort_values(["topology", "n_islands"]).reset_index(drop=True)
        labels = [f"{row['topology']}\nnp={row['n_islands']}"
                  for _, row in df.iterrows()]
        x = list(range(len(df)))
        bottom = [0.0] * len(df)
        for part, color in zip(parts, colors):
            vals = df[part].astype(float).tolist()
            ax.bar(x, vals, bottom=bottom, color=color, label=part)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_mpi_breakdown.png"), dpi=120)
    plt.close(fig)


def emit_table(strong, weak, base):
    out = ["# pso-cuda MPI scaling — M4 summary\n"]

    out.append("## Strong scaling (total N = 4096, D = 30, iters = 500)\n")
    t_single = float(base.loc[base["N"] == 4096, "total_ms"].iloc[0])
    out.append(f"Single-GPU baseline (pso_cuda, N=4096): "
               f"**total_ms = {t_single:.2f}**\n")
    s = merge_speedup(strong, t_single)
    cols = ["topology", "n_islands", "N", "eval_ms", "reduce_ms",
            "update_ms", "sync_ms", "total_ms",
            "speedup_vs_mpi1", "efficiency_vs_mpi1", "speedup_vs_single",
            "final_gbest"]
    out.append(s[cols].round(3).to_markdown(index=False))
    out.append("")

    out.append("## Weak scaling (N = 1024 per rank, D = 30, iters = 500)\n")
    t_single_w = float(base.loc[base["N"] == 1024, "total_ms"].iloc[0])
    out.append(f"Single-GPU baseline (pso_cuda, N=1024): "
               f"**total_ms = {t_single_w:.2f}**\n")
    w = merge_speedup(weak, t_single_w)
    out.append(w[cols].round(3).to_markdown(index=False))
    out.append("")

    out.append("## Notes\n")
    out.append("- `speedup_vs_mpi1` uses `pso_{topo} -np 1` as the baseline "
               "(MPI-native efficiency).")
    out.append("- `speedup_vs_single` uses the single-GPU `pso_cuda` binary "
               "as the baseline (honest absolute speedup).")
    out.append("- `sync_ms` is host wall time inside the on_sync callback, "
               "measured via `std::chrono::steady_clock`.")
    out.append("- `total_ms = eval_ms + reduce_ms + update_ms + sync_ms`.\n")

    with open(os.path.join(HERE, "table_mpi.md"), "w") as f:
        f.write("\n".join(out))


def main():
    strong = load("scaling_strong.csv")
    weak   = load("scaling_weak.csv")
    base   = load("scaling_baseline.csv")
    fig_scaling(strong, weak, base)
    fig_breakdown(strong, weak)
    emit_table(strong, weak, base)
    print("wrote bench/fig_mpi_scaling.png")
    print("wrote bench/fig_mpi_breakdown.png")
    print("wrote bench/table_mpi.md")


if __name__ == "__main__":
    main()
