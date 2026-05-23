#!/usr/bin/env python3
"""
analyze.py — consume bench CSVs and emit the figures/tables for the M3 report.

Inputs:
  bench/results.csv         (GPU sweep)
  bench/results_cpu.csv     (CPU baseline)
  bench/history_*.csv       (gbest-vs-iter curves)

Outputs:
  bench/fig_convergence.png
  bench/fig_speedup.png
  bench/fig_bw_gflops.png
  bench/fig_kernel_breakdown.png
  bench/table_summary.md     (markdown summary for the writeup)

Usage:
  cd pso-cuda
  python3 bench/analyze.py
"""

import glob
import os
import sys
import textwrap

from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Quadro RTX 6000 (Turing, sm_75) — used to compute "fraction of peak".
RTX6000_PEAK_BW_GBPS  = 672.0
RTX6000_PEAK_GFLOPS   = 16312.0   # FP32, ~16.3 TFLOPS

HERE = os.path.dirname(os.path.abspath(__file__))

# Evaluators to omit from figures/tables (still present in raw CSVs).
EXCLUDE_EVALUATORS = {"schaffer"}


def load() -> Tuple[pd.DataFrame, pd.DataFrame]:
    gpu = pd.read_csv(os.path.join(HERE, "results.csv"))
    cpu_path = os.path.join(HERE, "results_cpu.csv")
    cpu = pd.read_csv(cpu_path) if os.path.exists(cpu_path) else pd.DataFrame()
    # CPU baseline calls the evaluator "schaffer_f2"; GPU calls it "schaffer".
    if not cpu.empty:
        cpu["evaluator"] = cpu["evaluator"].replace({"schaffer_f2": "schaffer"})
    if EXCLUDE_EVALUATORS:
        gpu = gpu[~gpu["evaluator"].isin(EXCLUDE_EVALUATORS)].reset_index(drop=True)
        if not cpu.empty:
            cpu = cpu[~cpu["evaluator"].isin(EXCLUDE_EVALUATORS)].reset_index(drop=True)
    return gpu, cpu


def aggregate(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """Mean across seeds for each (impl, evaluator, N, D, iters)."""
    keys = ["impl", "evaluator", "N", "D", "iters"]
    return df.groupby(keys)[value_cols].mean().reset_index()


def speedup_table(gpu: pd.DataFrame, cpu: pd.DataFrame) -> pd.DataFrame:
    """ms-per-iter speedup, since CPU and GPU sweeps may use different iters."""
    if cpu.empty:
        return pd.DataFrame()

    g = gpu.copy()
    c = cpu.copy()
    g["ms_per_iter"] = g["total_ms"] / g["iters"]
    c["ms_per_iter"] = c["total_ms"] / c["iters"]

    g_agg = g.groupby(["evaluator", "N", "D"])["ms_per_iter"].mean().reset_index()
    c_agg = c.groupby(["evaluator", "N", "D"])["ms_per_iter"].mean().reset_index()

    j = c_agg.merge(g_agg, on=["evaluator", "N", "D"], suffixes=("_cpu", "_gpu"))
    j["speedup"] = j["ms_per_iter_cpu"] / j["ms_per_iter_gpu"]
    return j.sort_values(["evaluator", "N", "D"]).reset_index(drop=True)


def fig_convergence() -> None:
    files = sorted(glob.glob(os.path.join(HERE, "history_*.csv")))
    if not files:
        print("  [skip] no history_*.csv")
        return
    plt.figure(figsize=(7, 4.5))
    for path in files:
        name = os.path.basename(path).replace("history_", "").replace(".csv", "")
        if name in EXCLUDE_EVALUATORS:
            continue
        h = pd.read_csv(path)
        plt.plot(h["iter"], h["gbest"], label=name)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("gbest fitness (log)")
    plt.title("Convergence: gbest vs iteration  (N=1024, D=30, seed=1)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out = os.path.join(HERE, "fig_convergence.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  wrote {out}")


def fig_speedup(table: pd.DataFrame) -> None:
    if table.empty:
        print("  [skip] no CPU data")
        return
    plt.figure(figsize=(7, 4.5))
    for (evaluator, D), sub in table.groupby(["evaluator", "D"]):
        sub = sub.sort_values("N")
        plt.plot(sub["N"], sub["speedup"], marker="o",
                 label=f"{evaluator} D={D}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("N (swarm size)")
    plt.ylabel("speedup (CPU ms/iter ÷ GPU ms/iter)")
    plt.title("GPU speedup over CPU baseline")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out = os.path.join(HERE, "fig_speedup.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  wrote {out}")


def fig_bw_gflops(gpu_agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for evaluator, sub in gpu_agg.groupby("evaluator"):
        # Pick one D for the cross-N plot — D=30 is the canonical mid-size.
        s = sub[sub["D"] == 30].sort_values("N")
        if s.empty: continue
        axes[0].plot(s["N"], s["achieved_bw_gbps"], marker="o", label=evaluator)
        axes[1].plot(s["N"], s["achieved_gflops"],  marker="o", label=evaluator)

    axes[0].axhline(RTX6000_PEAK_BW_GBPS, ls="--", color="gray",
                    label=f"peak {RTX6000_PEAK_BW_GBPS:.0f} GB/s")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N"); axes[0].set_ylabel("achieved BW (GB/s)")
    axes[0].set_title("Memory bandwidth (D=30)")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    axes[1].axhline(RTX6000_PEAK_GFLOPS, ls="--", color="gray",
                    label=f"peak {RTX6000_PEAK_GFLOPS:.0f} GFLOPS")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("achieved GFLOPS")
    axes[1].set_title("Compute throughput (D=30)")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)

    out = os.path.join(HERE, "fig_bw_gflops.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  wrote {out}")


def fig_kernel_breakdown(gpu_agg: pd.DataFrame) -> None:
    # Pick a representative D, plot stacked bars per evaluator across N.
    sub = gpu_agg[gpu_agg["D"] == 30].copy()
    if sub.empty:
        print("  [skip] no D=30 rows"); return

    sub = sub.sort_values(["evaluator", "N"])
    labels = [f"{r.evaluator}\nN={r.N}" for r in sub.itertuples()]
    x = range(len(sub))

    plt.figure(figsize=(11, 4.5))
    plt.bar(x, sub["eval_ms"],   label="eval_ms")
    plt.bar(x, sub["reduce_ms"], bottom=sub["eval_ms"], label="reduce_ms")
    plt.bar(x, sub["update_ms"],
            bottom=sub["eval_ms"] + sub["reduce_ms"], label="update_ms")
    plt.xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("ms (mean across seeds)")
    plt.title("Per-kernel time breakdown (D=30, iters from sweep)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    out = os.path.join(HERE, "fig_kernel_breakdown.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  wrote {out}")


def write_summary_md(gpu_agg: pd.DataFrame, table: pd.DataFrame) -> None:
    lines = []
    lines.append("# pso-cuda — M3 initial performance summary\n")
    lines.append(f"Peak references (Quadro RTX 6000, sm_75): "
                 f"BW {RTX6000_PEAK_BW_GBPS:.0f} GB/s, FP32 {RTX6000_PEAK_GFLOPS:.0f} GFLOPS.\n")

    lines.append("## GPU performance (mean across seeds)\n")
    cols = ["evaluator", "N", "D", "iters",
            "eval_ms", "reduce_ms", "update_ms", "total_ms",
            "achieved_bw_gbps", "achieved_gflops", "final_gbest"]
    have = [c for c in cols if c in gpu_agg.columns]
    lines.append(gpu_agg[have].round(3).to_markdown(index=False))
    lines.append("")

    if not table.empty:
        lines.append("## Speedup vs CPU baseline (ms/iter ratio)\n")
        t = table.copy().round(4)
        lines.append(t.to_markdown(index=False))
        lines.append("")

        # Headline numbers.
        best = table.loc[table["speedup"].idxmax()]
        lines.append(f"**Headline speedup:** {best['speedup']:.1f}× on "
                     f"{best['evaluator']} N={int(best['N'])} D={int(best['D'])}.\n")
    else:
        lines.append("## Speedup vs CPU baseline\n")
        lines.append("_no CPU results found in bench/results_cpu.csv_\n")

    # Bound diagnosis.
    lines.append("## Bound analysis\n")
    g30 = gpu_agg[gpu_agg["D"] == 30]
    if not g30.empty:
        max_bw = g30["achieved_bw_gbps"].max()
        max_gf = g30["achieved_gflops"].max()
        bw_pct = 100 * max_bw / RTX6000_PEAK_BW_GBPS
        gf_pct = 100 * max_gf / RTX6000_PEAK_GFLOPS
        lines.append(textwrap.dedent(f"""\
            At D=30, best observed achieved bandwidth was {max_bw:.1f} GB/s
            ({bw_pct:.1f}% of {RTX6000_PEAK_BW_GBPS:.0f} GB/s peak) and best observed
            achieved compute was {max_gf:.1f} GFLOPS ({gf_pct:.1f}% of
            {RTX6000_PEAK_GFLOPS:.0f} GFLOPS peak).
        """))
        if bw_pct < 20 and gf_pct < 20:
            lines.append("Both BW and FLOPS are well below peak — the kernel is "
                         "likely **launch-/latency-bound** at these sizes, dominated "
                         "by kernel-launch overhead and per-iter sync (commit_gbest "
                         "is a single-thread kernel + cuRAND state traffic).")
        elif bw_pct > gf_pct * 2:
            lines.append("BW utilization dominates FLOPS utilization — the kernel "
                         "is **memory-bound** at these sizes.")
        elif gf_pct > bw_pct * 2:
            lines.append("FLOPS utilization dominates BW — the kernel is "
                         "**compute-bound** at these sizes.")
        else:
            lines.append("BW and FLOPS utilization are comparable — the kernel is "
                         "**balanced/roofline-mixed** at these sizes.")

    out = os.path.join(HERE, "table_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out}")


def main() -> int:
    gpu, cpu = load()
    if gpu.empty:
        print("error: bench/results.csv is empty. Run ./experiment.sh first.",
              file=sys.stderr)
        return 1

    gpu_value_cols = ["eval_ms", "reduce_ms", "update_ms", "total_ms",
                      "final_gbest", "achieved_bw_gbps", "achieved_gflops"]
    gpu_agg = aggregate(gpu, [c for c in gpu_value_cols if c in gpu.columns])

    table = speedup_table(gpu, cpu)

    print(">>> writing figures")
    fig_convergence()
    fig_speedup(table)
    fig_bw_gflops(gpu_agg)
    fig_kernel_breakdown(gpu_agg)
    write_summary_md(gpu_agg, table)

    print("\ndone. Open bench/table_summary.md and the four .png files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
