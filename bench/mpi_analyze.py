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


MPI_CSV_COLS = [
    "topology", "evaluator", "n_islands", "N", "D", "iters", "seed",
    "eval_ms", "reduce_ms", "update_ms", "sync_ms", "total_ms", "final_gbest",
]


def load(name, required=True, headerless_cols=None):
    """Load a CSV. If headerless_cols is given, use it as the column names
    (the MPI mains' --csv_path writer doesn't emit a header)."""
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        if required:
            sys.exit(f"missing: {path}")
        return None
    if headerless_cols is not None:
        return pd.read_csv(path, header=None, names=headerless_cols)
    return pd.read_csv(path)


def compute_ms(df):
    """Compute-only time: eval + reduce + update (excludes sync)."""
    return df["eval_ms"] + df["reduce_ms"] + df["update_ms"]


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
    """Build and return the markdown lines for the M4 base scaling tables.
    Caller writes them out (possibly after appending sweep tables)."""
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
    return out


#
# ============================================================================
# Phase E sweep figures (M4 follow-up)
# ============================================================================
#

def fig_sweep_N(sweep_N, baseline_N):
    """Two panels, log-x in N.
    Left:  ms breakdown (sync_ms vs compute_ms) per topology, plus pso_cuda total.
    Right: sync_ms / compute_ms ratio; horizontal ref line at y=1 (crossover).
    """
    s = sweep_N.sort_values(["topology", "N"]).reset_index(drop=True)
    s["compute_ms"] = compute_ms(s)
    s["ratio"] = s["sync_ms"] / s["compute_ms"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
    for topo, g in s.groupby("topology"):
        ax_l.plot(g["N"], g["sync_ms"],    marker="o", label=f"{topo} sync_ms")
        ax_l.plot(g["N"], g["compute_ms"], marker="s", linestyle="--",
                  label=f"{topo} compute_ms")

    # pso_cuda single-GPU total_ms as reference horizontal line at each N.
    b = baseline_N.sort_values("N").reset_index(drop=True)
    ax_l.plot(b["N"], b["total_ms"], marker="x", color="black",
              label="pso_cuda total_ms (single GPU)")

    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("N (particles per island)")
    ax_l.set_ylabel("time (ms)")
    ax_l.set_title("ms breakdown vs N (ranks=4, sync=10, iters=500)")
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3, which="both")

    for topo, g in s.groupby("topology"):
        ax_r.plot(g["N"], g["ratio"], marker="o", label=f"{topo}")
    ax_r.axhline(1.0, color="gray", linestyle=":",
                 label="sync = compute (crossover)")
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("N (particles per island)")
    ax_r.set_ylabel("sync_ms / compute_ms")
    ax_r.set_title("sync amortization (lower = MPI more competitive)")
    ax_r.legend(fontsize=8)
    ax_r.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_sweep_N.png"), dpi=120)
    plt.close(fig)


def fig_sweep_sync(sweep_sync):
    """Twin axes. x = sync_interval (log).
    Left y: sync_ms mean ± std (per topology, shaded band).
    Right y: final_gbest mean ± std (per topology, dashed).
    Pareto: fewer syncs = cheaper but worse convergence.
    """
    grp = sweep_sync.groupby(["topology", "sync_interval"]).agg(
        sync_ms_mean=("sync_ms", "mean"),
        sync_ms_std=("sync_ms", "std"),
        gbest_mean=("final_gbest", "mean"),
        gbest_std=("final_gbest", "std"),
    ).reset_index().sort_values(["topology", "sync_interval"])

    fig, ax_l = plt.subplots(figsize=(8, 4.5))
    ax_r = ax_l.twinx()

    colors = {"ring": "#4C72B0", "fc": "#C44E52"}
    for topo, g in grp.groupby("topology"):
        c = colors.get(topo, "gray")
        ax_l.plot(g["sync_interval"], g["sync_ms_mean"],
                  marker="o", color=c, label=f"{topo} sync_ms")
        ax_l.fill_between(g["sync_interval"],
                          g["sync_ms_mean"] - g["sync_ms_std"],
                          g["sync_ms_mean"] + g["sync_ms_std"],
                          color=c, alpha=0.15)
        ax_r.plot(g["sync_interval"], g["gbest_mean"],
                  marker="s", linestyle="--", color=c,
                  label=f"{topo} final_gbest")
        ax_r.fill_between(g["sync_interval"],
                          g["gbest_mean"] - g["gbest_std"],
                          g["gbest_mean"] + g["gbest_std"],
                          color=c, alpha=0.10)

    ax_l.set_xscale("log")
    ax_l.set_xlabel("sync_interval (iterations between syncs)")
    ax_l.set_ylabel("sync_ms (solid)")
    ax_r.set_ylabel("final_gbest (dashed, lower = better)")
    ax_l.set_title("Pareto: sync cost vs convergence (5 seeds, ranks=4, N=1024)")
    ax_l.grid(alpha=0.3, which="both")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r,
                fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_sweep_sync.png"), dpi=120)
    plt.close(fig)


def fig_sweep_NxRxD(matrix, baseline):
    """3-panel row, one per D. Each panel: log-x N, log-y total_ms,
    one line per (ranks, topology). Horizontal y=60000 ms reference.
    Plus dashed pso_cuda single-GPU at the matching D.
    """
    ds = sorted(matrix["D"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 4.5),
                             sharey=True)
    if len(ds) == 1:
        axes = [axes]

    rank_colors = {1: "#888888", 4: "#4C72B0", 16: "#C44E52"}
    topo_styles = {"ring": "-", "fc": "--"}

    for ax, D in zip(axes, ds):
        sub = matrix[matrix["D"] == D].sort_values("N")
        for (ranks, topo), g in sub.groupby(["n_islands", "topology"]):
            ax.plot(g["N"], g["total_ms"],
                    marker="o",
                    color=rank_colors.get(int(ranks), "black"),
                    linestyle=topo_styles.get(topo, "-"),
                    label=f"{topo} np={ranks}")
        if baseline is not None and "D" in baseline.columns:
            b = baseline[baseline["D"] == D].sort_values("N")
            if not b.empty:
                ax.plot(b["N"], b["total_ms"],
                        marker="x", color="black", linestyle=":",
                        label="pso_cuda (1 GPU)")
        ax.axhline(60000, color="red", linestyle=":", alpha=0.5,
                   label="60 s ceiling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("per-rank N")
        ax.set_title(f"D = {D}")
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0]:
            ax.set_ylabel("total_ms (log)")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Phase G — total_ms vs N at sync=25, m = max(5, N/100)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_sweep_NxRxD.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_sweep_largeN_scaling(strong, weak, baseline_largeN):
    """Same shape as fig_mpi_scaling.png but for the large-N sweeps.
    Two panels: speedup vs ranks, efficiency vs ranks. Both topologies.
    """
    t_single = float(baseline_largeN.loc[
        baseline_largeN["N"] == 16384, "total_ms"].iloc[0])
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
    ax_sp.set_title("strong scaling speedup at N_total = 16384")
    ax_sp.legend(fontsize=8)
    ax_sp.grid(alpha=0.3)

    ax_eff.axhline(1.0, color="gray", linestyle=":", label="ideal")
    ax_eff.set_xlabel("number of ranks (p)")
    ax_eff.set_ylabel("efficiency = speedup / p")
    ax_eff.set_title("strong scaling efficiency at N_total = 16384")
    ax_eff.legend(fontsize=8)
    ax_eff.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_sweep_largeN_scaling.png"), dpi=120)
    plt.close(fig)


def emit_sweep_tables(out_lines, sweep_N, baseline_N,
                      sweep_sync, sweep_strong, sweep_weak, baseline_largeN):
    """Append the three sweep summary tables to the markdown out_lines list."""
    # --- N sweep ---
    out_lines.append("## N sweep — sync amortization (ranks=4, sync=10, iters=500)\n")
    s = sweep_N.sort_values(["topology", "N"]).reset_index(drop=True).copy()
    s["compute_ms"] = compute_ms(s)
    s["sync_ratio"] = (s["sync_ms"] / s["compute_ms"]).round(3)
    b = baseline_N.sort_values("N").reset_index(drop=True)
    s = s.merge(b[["N", "total_ms"]].rename(
        columns={"total_ms": "pso_cuda_total_ms"}), on="N", how="left")
    s["speedup_vs_single"] = (s["pso_cuda_total_ms"] / s["total_ms"]).round(3)
    cols = ["topology", "N", "compute_ms", "sync_ms", "total_ms",
            "pso_cuda_total_ms", "sync_ratio",
            "speedup_vs_single", "final_gbest"]
    out_lines.append(s[cols].round(3).to_markdown(index=False))
    out_lines.append("")

    # --- sync_interval Pareto ---
    out_lines.append("## sync_interval Pareto (5 seeds, ranks=4, N=1024, iters=500)\n")
    grp = sweep_sync.groupby(["topology", "sync_interval"]).agg(
        sync_ms_mean=("sync_ms", "mean"),
        sync_ms_std=("sync_ms", "std"),
        total_ms_mean=("total_ms", "mean"),
        gbest_mean=("final_gbest", "mean"),
        gbest_std=("final_gbest", "std"),
    ).reset_index().sort_values(["topology", "sync_interval"])
    out_lines.append(grp.round(3).to_markdown(index=False))
    out_lines.append("")

    # --- large-N strong + weak ---
    out_lines.append("## Strong scaling at N_total = 16384\n")
    t_single = float(baseline_largeN.loc[
        baseline_largeN["N"] == 16384, "total_ms"].iloc[0])
    out_lines.append(
        f"Single-GPU baseline (pso_cuda, N=16384): **total_ms = {t_single:.2f}**\n")
    cols_sc = ["topology", "n_islands", "N", "eval_ms", "reduce_ms",
               "update_ms", "sync_ms", "total_ms",
               "speedup_vs_mpi1", "efficiency_vs_mpi1", "speedup_vs_single",
               "final_gbest"]
    out_lines.append(
        merge_speedup(sweep_strong, t_single)[cols_sc].round(3).to_markdown(index=False))
    out_lines.append("")

    # (Sweep 3b weak table emits below — kept as-is)

    out_lines.append("## Weak scaling at per-rank N = 16384\n")
    # No matching single-GPU baseline at total N=16384*p; use the np=1 row.
    t_mpi1 = float(sweep_weak.loc[
        (sweep_weak["topology"] == "ring") & (sweep_weak["n_islands"] == 1),
        "total_ms"].iloc[0])
    out_lines.append(
        merge_speedup(sweep_weak, t_mpi1)[cols_sc].round(3).to_markdown(index=False))
    out_lines.append("")


def fig_largeN_strong_weak(strong, weak, baseline):
    """2x2 grid summarizing Phase H large-N strong + weak scaling.
       top-left:  strong total_ms vs ranks
       top-right: strong speedup vs ranks (two baselines per topology)
       bottom-left:  weak total_ms vs ranks
       bottom-right: stacked-bar breakdown of eval/reduce/update/sync per
                     (ranks, topology) for both strong and weak.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    topo_styles = {"ring": "-", "fc": "--"}
    topo_colors = {"ring": "#4C72B0", "fc": "#C44E52"}

    # --- Strong scaling: total_ms vs ranks
    ax = axes[0, 0]
    if strong is not None and not strong.empty:
        for topo, g in strong.sort_values("n_islands").groupby("topology"):
            ax.plot(g["n_islands"], g["total_ms"],
                    marker="o",
                    color=topo_colors.get(topo, "black"),
                    linestyle=topo_styles.get(topo, "-"),
                    label=f"{topo}")
    ax.set_xlabel("ranks")
    ax.set_ylabel("total_ms")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Strong scaling: total_ms (N_total=8M, D=100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- Strong scaling: speedup vs ranks (two baselines)
    ax = axes[0, 1]
    if strong is not None and not strong.empty:
        t_single = None
        if baseline is not None and not baseline.empty:
            row = baseline[baseline["N"] == 8388608]
            if not row.empty:
                t_single = float(row["total_ms"].iloc[0])
        for topo, g in strong.sort_values("n_islands").groupby("topology"):
            t_mpi1 = g.loc[g["n_islands"] == 1, "total_ms"]
            if not t_mpi1.empty:
                ax.plot(g["n_islands"], float(t_mpi1.iloc[0]) / g["total_ms"],
                        marker="o",
                        color=topo_colors.get(topo, "black"),
                        linestyle=topo_styles.get(topo, "-"),
                        label=f"{topo} vs pso_{topo} -np 1")
            if t_single is not None:
                ax.plot(g["n_islands"], t_single / g["total_ms"],
                        marker="s",
                        color=topo_colors.get(topo, "black"),
                        linestyle=":",
                        alpha=0.7,
                        label=f"{topo} vs pso_cuda single-GPU")
        xs = sorted(strong["n_islands"].unique())
        ax.plot(xs, xs, color="gray", linestyle=":", alpha=0.5,
                label="ideal y = p")
    ax.set_xlabel("ranks")
    ax.set_ylabel("speedup")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Strong scaling: speedup")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # --- Weak scaling: total_ms vs ranks
    ax = axes[1, 0]
    if weak is not None and not weak.empty:
        for topo, g in weak.sort_values("n_islands").groupby("topology"):
            ax.plot(g["n_islands"], g["total_ms"],
                    marker="o",
                    color=topo_colors.get(topo, "black"),
                    linestyle=topo_styles.get(topo, "-"),
                    label=f"{topo}")
    ax.set_xlabel("ranks")
    ax.set_ylabel("total_ms")
    ax.set_xscale("log", base=2)
    ax.set_title("Weak scaling: total_ms (per-rank N=8M, D=100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- Stacked-bar breakdown
    ax = axes[1, 1]
    parts = ["eval_ms", "reduce_ms", "update_ms", "sync_ms"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    rows = []
    if strong is not None and not strong.empty:
        for _, r in strong.sort_values(["topology", "n_islands"]).iterrows():
            rows.append(("S", r))
    if weak is not None and not weak.empty:
        for _, r in weak.sort_values(["topology", "n_islands"]).iterrows():
            rows.append(("W", r))
    labels = [f"{tag}-{r['topology']}\nnp={int(r['n_islands'])}"
              for tag, r in rows]
    x = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for part, color in zip(parts, colors):
        vals = [float(r[part]) for _, r in rows]
        ax.bar(x, vals, bottom=bottom, color=color, label=part)
        bottom = [b + v for b, v in zip(bottom, vals)]
    if x:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("time (ms)")
    ax.set_title("Comm/compute breakdown (S = strong, W = weak)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Phase H — large-N strong + weak scaling (D=100, sync=25, m=N/100)",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_largeN_strong_weak.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def emit_largeN_tables(out_lines, strong, weak, baseline):
    """Append Phase H strong + weak tables to the markdown."""
    out_lines.append("\n---\n\n# Phase H — Large-N strong + weak scaling\n")
    out_lines.append("`D = 100`, `sync_interval = 25`, `migrate = max(5, N/100)`, "
                     "`iters = 500`, `evaluator = rastrigin`.\n")

    if baseline is not None and not baseline.empty:
        b = baseline[baseline["N"] == 8388608]
        if not b.empty:
            t = float(b["total_ms"].iloc[0])
            g = float(b["final_gbest"].iloc[0])
            out_lines.append(
                f"Single-GPU baseline (`pso_cuda --N 8388608 --D 100`): "
                f"**total_ms = {t:.1f}**, final_gbest = {g:.3f}.\n")

    if strong is not None and not strong.empty:
        out_lines.append("## Strong scaling (N_total = 8M)\n")
        s = strong.copy().sort_values(["topology", "n_islands"])
        for topo, gtopo in s.groupby("topology"):
            t_mpi1 = gtopo.loc[gtopo["n_islands"] == 1, "total_ms"]
            if not t_mpi1.empty:
                gtopo = gtopo.copy()
                gtopo["speedup_vs_mpi1"] = (float(t_mpi1.iloc[0])
                                            / gtopo["total_ms"]).round(3)
                gtopo["efficiency"] = (gtopo["speedup_vs_mpi1"]
                                       / gtopo["n_islands"]).round(3)
                s.loc[gtopo.index, "speedup_vs_mpi1"] = gtopo["speedup_vs_mpi1"]
                s.loc[gtopo.index, "efficiency"] = gtopo["efficiency"]
        cols = ["topology", "n_islands", "N", "eval_ms", "reduce_ms",
                "update_ms", "sync_ms", "total_ms",
                "speedup_vs_mpi1", "efficiency", "final_gbest"]
        cols = [c for c in cols if c in s.columns]
        out_lines.append(s[cols].round(3).to_markdown(index=False))
        out_lines.append("")

    if weak is not None and not weak.empty:
        out_lines.append("## Weak scaling (per-rank N = 8M)\n")
        w = weak.copy().sort_values(["topology", "n_islands"])
        for topo, gtopo in w.groupby("topology"):
            t_mpi1 = gtopo.loc[gtopo["n_islands"] == 1, "total_ms"]
            if not t_mpi1.empty:
                gtopo = gtopo.copy()
                gtopo["efficiency"] = (float(t_mpi1.iloc[0])
                                       / gtopo["total_ms"]).round(3)
                w.loc[gtopo.index, "efficiency"] = gtopo["efficiency"]
        cols = ["topology", "n_islands", "N", "eval_ms", "reduce_ms",
                "update_ms", "sync_ms", "total_ms",
                "efficiency", "final_gbest"]
        cols = [c for c in cols if c in w.columns]
        out_lines.append(w[cols].round(3).to_markdown(index=False))
        out_lines.append("")


def emit_NxRxD_table(out_lines, matrix, baseline, levy):
    """Pivot indexed by (D, ranks, topology, N) with compute / sync / total /
    sync_ratio / final_gbest. Plus a single-GPU baseline column merged by (D, N)
    for context, and the Levy sanity rows separately.
    """
    out_lines.append("\n---\n\n# Phase G — N x ranks x D matrix\n")
    out_lines.append("`sync_interval = 25`, `migrate = max(5, N/100)`, "
                     "`iters = 500`, `evaluator = rastrigin`.\n")

    m = matrix.copy()
    m["compute_ms"] = compute_ms(m)
    m["sync_ratio"] = (m["sync_ms"] / m["compute_ms"]).round(3)

    if baseline is not None and not baseline.empty:
        b = baseline.rename(columns={"total_ms": "pso_cuda_total_ms"})
        m = m.merge(b[["D", "N", "pso_cuda_total_ms"]],
                    on=["D", "N"], how="left")
        m["speedup_vs_single"] = (
            m["pso_cuda_total_ms"] / m["total_ms"]).round(3)
    else:
        m["pso_cuda_total_ms"] = None
        m["speedup_vs_single"] = None

    m = m.sort_values(["D", "n_islands", "topology", "N"]).reset_index(drop=True)
    cols = ["D", "n_islands", "topology", "N", "compute_ms", "sync_ms",
            "total_ms", "sync_ratio", "pso_cuda_total_ms", "speedup_vs_single",
            "final_gbest"]
    out_lines.append(m[cols].round(3).to_markdown(index=False))
    out_lines.append("")

    if levy is not None and not levy.empty:
        out_lines.append("## Levy sanity (np=4 ring, N=524288)\n")
        out_lines.append(
            levy[["D", "topology", "n_islands", "N", "total_ms", "final_gbest"]]
            .round(6).to_markdown(index=False))
        out_lines.append("")


def main():
    # M4 base scaling — always present.
    strong = load("scaling_strong.csv")
    weak   = load("scaling_weak.csv")
    base   = load("scaling_baseline.csv")
    fig_scaling(strong, weak, base)
    fig_breakdown(strong, weak)
    table_lines = emit_table(strong, weak, base)
    print("wrote bench/fig_mpi_scaling.png")
    print("wrote bench/fig_mpi_breakdown.png")

    # Phase E sweeps — optional. Generate figures and append table sections
    # only when the corresponding CSVs are present.
    # The MPI mains' --csv_path writer doesn't emit a header; provide one.
    sweep_N      = load("sweep_N.csv",            required=False,
                        headerless_cols=MPI_CSV_COLS)
    base_N       = load("sweep_N_baseline.csv",   required=False)
    sweep_sync   = load("sweep_sync.csv",         required=False)
    sweep_strong = load("sweep_strong_largeN.csv", required=False,
                        headerless_cols=MPI_CSV_COLS)
    sweep_weak   = load("sweep_weak_largeN.csv",   required=False,
                        headerless_cols=MPI_CSV_COLS)
    base_largeN  = load("sweep_largeN_baseline.csv", required=False)

    if sweep_N is not None and base_N is not None:
        fig_sweep_N(sweep_N, base_N)
        print("wrote bench/fig_sweep_N.png")

    if sweep_sync is not None:
        fig_sweep_sync(sweep_sync)
        print("wrote bench/fig_sweep_sync.png")

    if (sweep_strong is not None and sweep_weak is not None
            and base_largeN is not None):
        fig_sweep_largeN_scaling(sweep_strong, sweep_weak, base_largeN)
        print("wrote bench/fig_sweep_largeN_scaling.png")

    if all(x is not None for x in (sweep_N, base_N, sweep_sync,
                                   sweep_strong, sweep_weak, base_largeN)):
        table_lines.append("\n---\n\n# Phase E sweeps\n")
        emit_sweep_tables(table_lines, sweep_N, base_N, sweep_sync,
                          sweep_strong, sweep_weak, base_largeN)

    # Phase G — N x ranks x D matrix.
    nxr_matrix = load("sweep_NxRxD.csv",          required=False,
                      headerless_cols=MPI_CSV_COLS)
    nxr_base   = load("sweep_NxRxD_baseline.csv", required=False)
    nxr_levy   = load("sweep_NxRxD_levy.csv",     required=False,
                      headerless_cols=MPI_CSV_COLS)

    if nxr_matrix is not None and not nxr_matrix.empty:
        fig_sweep_NxRxD(nxr_matrix,
                        nxr_base if nxr_base is not None else pd.DataFrame())
        print("wrote bench/fig_sweep_NxRxD.png")
        emit_NxRxD_table(table_lines, nxr_matrix, nxr_base, nxr_levy)

    # Phase H — large-N strong + weak scaling.
    largeN_strong = load("sweep_largeN_strong.csv",          required=False,
                         headerless_cols=MPI_CSV_COLS)
    largeN_weak   = load("sweep_largeN_weak.csv",            required=False,
                         headerless_cols=MPI_CSV_COLS)
    largeN_base   = load("sweep_largeN_strong_baseline.csv", required=False)

    if (largeN_strong is not None and not largeN_strong.empty) \
            or (largeN_weak is not None and not largeN_weak.empty):
        fig_largeN_strong_weak(largeN_strong, largeN_weak, largeN_base)
        print("wrote bench/fig_largeN_strong_weak.png")
        emit_largeN_tables(table_lines, largeN_strong, largeN_weak, largeN_base)

    with open(os.path.join(HERE, "table_mpi.md"), "w") as f:
        f.write("\n".join(table_lines))
    print("wrote bench/table_mpi.md")


if __name__ == "__main__":
    main()
