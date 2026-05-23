#!/usr/bin/env bash
# experiment.sh — initial perf measurement sweep for the M3 progress report.
#
# Produces:
#   bench/results.csv      — GPU rows (one per (eval, N, D, iters, seed))
#   bench/results_cpu.csv  — CPU baseline rows (subset of configs; CPU is slow)
#   bench/history_*.csv    — gbest-vs-iter curves for the convergence figure
#
# Usage:
#   ./experiment.sh           # full sweep (~5 min on RTX 6000)
#   ./experiment.sh --quick   # ~30s smoke sweep, fewer configs

set -euo pipefail

cd "$(dirname "$0")"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

# ---- build -------------------------------------------------------------------
echo ">>> building gpu + cpu binaries"
make -s all cpu

GPU=./build/pso_cuda
CPU=./build/pso_cpu

mkdir -p bench
GPU_CSV=bench/results.csv
CPU_CSV=bench/results_cpu.csv

# Back up any previous run so we don't pollute it.
ts=$(date +%Y%m%d_%H%M%S)
[[ -f "$GPU_CSV" ]] && mv "$GPU_CSV" "bench/results.${ts}.bak.csv" && \
    echo ">>> backed up old GPU CSV to bench/results.${ts}.bak.csv"
[[ -f "$CPU_CSV" ]] && mv "$CPU_CSV" "bench/results_cpu.${ts}.bak.csv" && \
    echo ">>> backed up old CPU CSV to bench/results_cpu.${ts}.bak.csv"

# ---- sweep configuration -----------------------------------------------------
EVALUATORS=(rastrigin levy schaffer)

if [[ $QUICK -eq 1 ]]; then
    GPU_NS=(256 4096)
    GPU_DS=(10 30)
    GPU_ITERS=500
    GPU_SEEDS=(1 2 3)

    CPU_EVALS=(rastrigin levy)
    CPU_NS=(256)
    CPU_DS=(10 30)
    CPU_ITERS=500
    CPU_SEEDS=(1 2)
else
    GPU_NS=(256 1024 4096 16384)
    GPU_DS=(10 30 100)
    GPU_ITERS=1000
    GPU_SEEDS=(1 2 3 4 5)

    # CPU is O(N*D*iters) per iter and single-threaded — keep configs small.
    CPU_EVALS=(rastrigin levy)
    CPU_NS=(256 1024)
    CPU_DS=(10 30)
    CPU_ITERS=500
    CPU_SEEDS=(1 2 3)
fi

# ---- GPU sweep ---------------------------------------------------------------
# Schaffer F2 is mathematically defined only at D=2 (returns INFINITY otherwise),
# so it gets its own dim list. Rastrigin/Levy sweep the full D range.
gpu_total=0
for eval in "${EVALUATORS[@]}"; do
  if [[ "$eval" == "schaffer" ]]; then ds=(2); else ds=("${GPU_DS[@]}"); fi
  gpu_total=$((gpu_total + ${#GPU_NS[@]} * ${#ds[@]} * ${#GPU_SEEDS[@]}))
done
i=0
echo ">>> GPU sweep: $gpu_total runs"
for eval in "${EVALUATORS[@]}"; do
  if [[ "$eval" == "schaffer" ]]; then ds=(2); else ds=("${GPU_DS[@]}"); fi
  for N in "${GPU_NS[@]}"; do
    for D in "${ds[@]}"; do
      for seed in "${GPU_SEEDS[@]}"; do
        i=$((i + 1))
        printf "  [%3d/%3d] gpu  %-9s N=%-5d D=%-3d iters=%-4d seed=%d\n" \
            "$i" "$gpu_total" "$eval" "$N" "$D" "$GPU_ITERS" "$seed"
        "$GPU" --evaluator "$eval" --N "$N" --D "$D" \
               --iters "$GPU_ITERS" --seed "$seed" \
               --csv_path "$GPU_CSV" > /dev/null
      done
    done
  done
done

# ---- CPU sweep ---------------------------------------------------------------
# pso_cpu prints one CSV row to stdout per run; concat into CPU_CSV.
# Schema: impl,evaluator,N,D,iters,seed,total_ms,best_value,best_position0
CPU_HEADER="impl,evaluator,N,D,iters,seed,total_ms,best_value,best_position0"
echo "$CPU_HEADER" > "$CPU_CSV"

cpu_total=0
for eval in "${CPU_EVALS[@]}"; do
  if [[ "$eval" == "schaffer" ]]; then cds=(2); else cds=("${CPU_DS[@]}"); fi
  cpu_total=$((cpu_total + ${#CPU_NS[@]} * ${#cds[@]} * ${#CPU_SEEDS[@]}))
done
i=0
echo ">>> CPU sweep: $cpu_total runs"
for eval in "${CPU_EVALS[@]}"; do
  # CPU baseline uses "schaffer_f2" while GPU uses "schaffer". Translate.
  cpu_eval="$eval"
  [[ "$eval" == "schaffer" ]] && cpu_eval="schaffer_f2"
  if [[ "$eval" == "schaffer" ]]; then cds=(2); else cds=("${CPU_DS[@]}"); fi
  for N in "${CPU_NS[@]}"; do
    for D in "${cds[@]}"; do
      for seed in "${CPU_SEEDS[@]}"; do
        i=$((i + 1))
        printf "  [%3d/%3d] cpu  %-9s N=%-5d D=%-3d iters=%-4d seed=%d\n" \
            "$i" "$cpu_total" "$eval" "$N" "$D" "$CPU_ITERS" "$seed"
        # CPU baseline writes the header on every run; skip it on append.
        "$CPU" "$N" "$D" "$CPU_ITERS" "$seed" "$cpu_eval" \
            | tail -n +2 >> "$CPU_CSV"
      done
    done
  done
done

# ---- convergence-curve dumps -------------------------------------------------
# One per evaluator, fixed config, single seed — enough for the convergence
# figure in the report.
echo ">>> dumping convergence histories"
for eval in "${EVALUATORS[@]}"; do
  out="bench/history_${eval}.csv"
  d=30; [[ "$eval" == "schaffer" ]] && d=2
  printf "  history -> %s (D=%d)\n" "$out" "$d"
  "$GPU" --evaluator "$eval" --N 1024 --D "$d" --iters 1000 --seed 1 \
         --history "$out" > /dev/null
done

# ---- summary -----------------------------------------------------------------
echo
echo ">>> done."
echo "    GPU rows: $(($(wc -l < "$GPU_CSV") - 1))  ($GPU_CSV)"
echo "    CPU rows: $(($(wc -l < "$CPU_CSV") - 1))  ($CPU_CSV)"
echo "    histories: $(ls bench/history_*.csv 2>/dev/null | wc -l | tr -d ' ')"
echo
echo ">>> sample GPU rows:"
head -1 "$GPU_CSV"
head -6 "$GPU_CSV" | tail -5
echo
echo ">>> sample CPU rows:"
head -4 "$CPU_CSV"
