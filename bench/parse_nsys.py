#!/usr/bin/env python3
"""Parse bench/nsys_summary_D*_N*_np*.txt files into a tidy CSV and two
markdown tables for §3.7 of M4_REPORT.md.

Input: one nsys_summary_D{D}_N{N}_np{ranks}.txt per cell. Each file has
two sections produced by `nsys stats --report cuda_api_sum,cuda_gpu_kern_sum`:

    ** CUDA API Summary (cuda_api_sum):
        <column header>
        --- separator ---
        <data rows>          # blank line terminates

    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
        <column header>
        --- separator ---
        <data rows>          # blank line terminates

Each data row: 8 numeric columns (Time%, Total Time ns, Num Calls / Instances,
Avg, Med, Min, Max, StdDev) followed by the name. Name may contain whitespace
(C++ kernel signatures), so we treat tokens 9+ as the name.

Outputs:
- bench/nsight_matrix.csv  (long format: D, N, ranks, category, name, total_ms, percent)
- bench/nsight_tables.md   (two markdown tables: API + kernel breakdown)
"""

import os
import re
import glob
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_GLOB = os.path.join(HERE, "nsys_summary_D*_N*_np*.txt")
CSV_OUT      = os.path.join(HERE, "nsight_matrix.csv")
MD_OUT       = os.path.join(HERE, "nsight_tables.md")

# Substring patterns mapped to canonical short names used as table columns.
API_CANONICAL = [
    ("cudaMemcpyFromSymbol", "cudaMemcpyFromSymbol"),
    ("cudaMemcpy",           "cudaMemcpy"),
    ("cudaDeviceSynchronize","cudaDeviceSync"),
    ("cudaLaunchKernel",     "cudaLaunchKernel"),
]
KERNEL_CANONICAL = [
    ("kernel_eval_and_pbest", "eval_and_pbest"),
    ("kernel_update",         "update"),
    ("kernel_draw_rng",       "draw_rng"),
    ("DeviceReduce",          "CUB_ArgMin"),       # both single-tile + multi-tile
    ("kernel_commit_gbest",   "commit_gbest"),
]

API_COLS    = ["cudaMemcpy", "cudaDeviceSync", "cudaLaunchKernel",
               "cudaMemcpyFromSymbol", "other"]
KERNEL_COLS = ["eval_and_pbest", "update", "draw_rng", "CUB_ArgMin",
               "commit_gbest", "other"]


def canonicalize(name: str, kind: str) -> str:
    """Map a raw nsys 'Name' column entry to one of our canonical columns."""
    table = API_CANONICAL if kind == "api" else KERNEL_CANONICAL
    for needle, canon in table:
        if needle in name:
            return canon
    return "other"


def parse_section(lines, start_idx: int):
    """Parse one nsys stats section starting at the `** ... Summary` header.

    Returns (dict {canonical: total_ms_summed}, end_idx) where end_idx is
    the line after the section.
    """
    # Skip the header line and the column-header + separator lines.
    i = start_idx + 1
    while i < len(lines) and not lines[i].lstrip().startswith("Time (%)"):
        i += 1
    if i >= len(lines):
        return {}, i
    # next line is the column header; the one after is the dash separator
    i += 2

    bucket = {}
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():       # blank line terminates the section
            break
        tokens = re.split(r"\s+", line.strip())
        if len(tokens) < 9:
            i += 1
            continue
        # tokens[0] = pct, tokens[1] = total time (ns), tokens[2] = calls,
        # tokens[3:8] = avg/med/min/max/stddev, tokens[8:] = name
        try:
            pct = float(tokens[0])
            total_ns = float(tokens[1].replace(",", ""))
        except ValueError:
            i += 1
            continue
        name = " ".join(tokens[8:])
        bucket.setdefault("_raw", []).append((name, total_ns / 1e6, pct))
        i += 1
    return bucket, i


def parse_file(path: str):
    """Parse one nsys_summary_*.txt file.

    Returns dict with keys 'api' and 'kernel', each a list of
    (name, total_ms, percent) tuples.
    """
    with open(path) as f:
        lines = f.readlines()

    out = {"api": [], "kernel": []}
    i = 0
    while i < len(lines):
        line = lines[i]
        if "CUDA API Summary" in line:
            bucket, i = parse_section(lines, i)
            out["api"] = bucket.get("_raw", [])
        elif "CUDA GPU Kernel Summary" in line:
            bucket, i = parse_section(lines, i)
            out["kernel"] = bucket.get("_raw", [])
        else:
            i += 1
    return out


def aggregate(entries, kind: str, columns: list):
    """Sum total_ms by canonical column for one cell."""
    sums = {col: 0.0 for col in columns}
    for name, ms, _pct in entries:
        sums[canonicalize(name, kind)] += ms
    return sums


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f} s"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def cells_from_summaries():
    """Discover (D, N, ranks, path) tuples from filenames."""
    pat = re.compile(r"nsys_summary_D(\d+)_N(\d+)_np(\d+)\.txt$")
    cells = []
    for path in sorted(glob.glob(SUMMARY_GLOB)):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        D, N, ranks = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells.append((D, N, ranks, path))
    return sorted(cells, key=lambda c: (c[0], c[1], c[2]))


def main():
    cells = cells_from_summaries()
    if not cells:
        sys.exit(f"no files matched {SUMMARY_GLOB}")

    # Build the long-format CSV and the table-shaped dict in one pass.
    csv_rows = []           # (D, N, ranks, category, name, total_ms, percent)
    api_table = []          # one row per cell, dict of canonical -> ms
    kernel_table = []

    for D, N, ranks, path in cells:
        parsed = parse_file(path)
        # Long-form CSV rows include the raw (name, ms, pct) so the file is
        # a complete record of what nsys reported.
        for name, ms, pct in parsed["api"]:
            csv_rows.append((D, N, ranks, "api", name, f"{ms:.4f}", f"{pct:.2f}"))
        for name, ms, pct in parsed["kernel"]:
            csv_rows.append((D, N, ranks, "kernel", name, f"{ms:.4f}", f"{pct:.2f}"))

        api_row    = aggregate(parsed["api"],    "api",    API_COLS)
        kernel_row = aggregate(parsed["kernel"], "kernel", KERNEL_COLS)
        api_row.update({"D": D, "N": N, "ranks": ranks})
        kernel_row.update({"D": D, "N": N, "ranks": ranks})
        api_table.append(api_row)
        kernel_table.append(kernel_row)

    # Write the CSV.
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "N", "ranks", "category", "name",
                    "total_ms", "percent"])
        w.writerows(csv_rows)
    print(f"wrote {CSV_OUT} ({len(csv_rows)} rows)")

    # Build the markdown tables.
    def fmt_N(n):
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1000:
            return f"{n // 1000}K"
        return str(n)

    def fmt_int(n):
        return f"{n:,}"

    def render(table, columns, title):
        lines = [f"### {title}\n"]
        header = ["D", "N", "np"] + columns + ["total"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in table:
            vals = [f"{row[c]:.1f}" if row[c] >= 1 else f"{row[c]:.2f}"
                    for c in columns]
            total = sum(row[c] for c in columns)
            total_fmt = (f"{total/1000:.2f} s" if total >= 1000
                         else f"{total:.1f}")
            lines.append("| " + " | ".join(
                [str(row["D"]), fmt_N(row["N"]), str(row["ranks"])]
                + vals + [total_fmt]) + " |")
        return "\n".join(lines)

    api_md = render(api_table, API_COLS,
                    "Table A — CUDA API breakdown (ms, rank 0)")
    kern_md = render(kernel_table, KERNEL_COLS,
                     "Table B — GPU kernel breakdown (ms, rank 0)")

    note = (
        "All cells: `pso_ring` rastrigin, iters=100, sync=25, "
        "m=max(5, N/100), seed=42. Numbers are wall-time milliseconds "
        "for rank 0 (representative; other ranks behave the same for ring). "
        "Cells that timed out at 90 s do not appear; check "
        "`bench/nsys_matrix_*.out` for which were dropped. The raw "
        "nsys stats output for each cell is in "
        "`bench/nsys_summary_D{D}_N{N}_np{ranks}.txt`.\n"
    )

    with open(MD_OUT, "w") as f:
        f.write("# Nsight matrix tables (rank 0, ring)\n\n")
        f.write(note + "\n")
        f.write(api_md + "\n\n")
        f.write(kern_md + "\n")
    print(f"wrote {MD_OUT}")

    # Echo the tables so the user can scroll-review.
    print("\n" + api_md + "\n\n" + kern_md)


if __name__ == "__main__":
    main()
