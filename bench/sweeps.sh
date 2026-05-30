#!/bin/bash -l
#SBATCH --job-name=pso_sweeps
#SBATCH --output=bench/sweeps_%j.out
#SBATCH --error=bench/sweeps_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=15

# Three parameter sweeps for the M4 follow-up (see plan Phase E):
#   1. N-sweep   — vary N, fix sync=10, ranks=4. Tests sync amortization.
#   2. sync-sweep — vary sync_interval, fix N=1024, ranks=4. Pareto front.
#   3. large-N strong + weak scaling — rank counts at N=16384.
#
# One warmup run per (N, topology) is discarded for sweeps 1 and 3.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

EVAL=rastrigin
D=30
ITERS=500
MIGRATE=5

run_mpi_csv() {
    # Append one MPI row to a --csv_path-compatible CSV via the binary's writer.
    local topo=$1 ranks=$2 N=$3 sync=$4 seed=$5 csv=$6
    mpirun -np "$ranks" "./build/pso_${topo}" \
        --evaluator "$EVAL" --N "$N" --D "$D" \
        --iters "$ITERS" --sync "$sync" --migrate "$MIGRATE" \
        --seed "$seed" --csv_path "$csv" > /dev/null
}

run_baseline_csv() {
    # Append one pso_cuda row to a single-GPU CSV via its writer.
    local N=$1 seed=$2 csv=$3
    ./build/pso_cuda \
        --evaluator "$EVAL" --N "$N" --D "$D" \
        --iters "$ITERS" --seed "$seed" --csv_path "$csv" > /dev/null
}

run_warmup() {
    # Same shape as run_mpi_csv but output discarded — flushes any cold-start cost.
    local topo=$1 ranks=$2 N=$3
    mpirun -np "$ranks" "./build/pso_${topo}" \
        --evaluator "$EVAL" --N "$N" --D "$D" \
        --iters 50 --sync 10 --migrate "$MIGRATE" \
        --seed 0 > /dev/null 2>&1
}

# ============================================================================
# Sweep 1 — N effect on amortization
# ============================================================================
echo "=== Sweep 1: N effect ==="
SWEEP_N=bench/sweep_N.csv
SWEEP_N_BASE=bench/sweep_N_baseline.csv
rm -f "$SWEEP_N" "$SWEEP_N_BASE"

for N in 1024 4096 16384 65536; do
    for TOPO in ring fc; do
        echo "  warmup: $TOPO N=$N"
        run_warmup "$TOPO" 4 "$N"
        echo "  measure: $TOPO N=$N ranks=4"
        run_mpi_csv "$TOPO" 4 "$N" 10 42 "$SWEEP_N"
    done
    echo "  baseline: pso_cuda N=$N"
    run_baseline_csv "$N" 42 "$SWEEP_N_BASE"
done

# ============================================================================
# Sweep 2 — sync_interval Pareto (custom CSV with sync_interval column)
# ============================================================================
echo ""
echo "=== Sweep 2: sync_interval Pareto ==="
SWEEP_SYNC=bench/sweep_sync.csv
echo "topology,sync_interval,seed,N,iters,eval_ms,reduce_ms,update_ms,sync_ms,total_ms,final_gbest" > "$SWEEP_SYNC"

for TOPO in ring fc; do
    echo "  warmup: $TOPO sync-sweep"
    run_warmup "$TOPO" 4 1024
    for SYNC in 1 5 10 25 50 100; do
        for SEED in 42 43 44 45 46; do
            out=$(mpirun -np 4 "./build/pso_${TOPO}" \
                    --evaluator "$EVAL" --N 1024 --D "$D" \
                    --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
                    --seed "$SEED" 2>&1)
            gbest=$(echo "$out" | awk -F'= ' '/^global best_value/ {print $2}')
            eval_ms=$(echo "$out" | awk -F'= ' '/^eval_ms/ {print $2}')
            red_ms=$(echo "$out" | awk -F'= ' '/^reduce_ms/ {print $2}')
            upd_ms=$(echo "$out" | awk -F'= ' '/^update_ms/ {print $2}')
            sync_ms=$(echo "$out" | awk -F'= ' '/^sync_ms/ {print $2}')
            tot_ms=$(echo "$out" | awk -F'= ' '/^total_ms/ {print $2}')
            echo "$TOPO,$SYNC,$SEED,1024,$ITERS,$eval_ms,$red_ms,$upd_ms,$sync_ms,$tot_ms,$gbest" >> "$SWEEP_SYNC"
        done
        echo "  done: $TOPO sync=$SYNC (5 seeds)"
    done
done

# ============================================================================
# Sweep 3a — strong scaling at N_total = 16384
# ============================================================================
echo ""
echo "=== Sweep 3a: strong scaling at N_total=16384 ==="
SWEEP_STRONG=bench/sweep_strong_largeN.csv
rm -f "$SWEEP_STRONG"

N_TOTAL=16384
for RANKS in 1 2 4; do
    N_PER=$(( N_TOTAL / RANKS ))
    for TOPO in ring fc; do
        echo "  measure: strong $TOPO ranks=$RANKS N=$N_PER"
        run_mpi_csv "$TOPO" "$RANKS" "$N_PER" 10 42 "$SWEEP_STRONG"
    done
done

# Single-GPU baseline at N=16384 for the speedup-vs-single comparison.
echo "  baseline: pso_cuda N=16384"
run_baseline_csv 16384 42 bench/sweep_largeN_baseline.csv

# ============================================================================
# Sweep 3b — weak scaling at per-rank N = 16384
# ============================================================================
echo ""
echo "=== Sweep 3b: weak scaling at per-rank N=16384 ==="
SWEEP_WEAK=bench/sweep_weak_largeN.csv
rm -f "$SWEEP_WEAK"

for RANKS in 1 2 4; do
    for TOPO in ring fc; do
        echo "  measure: weak $TOPO ranks=$RANKS N=16384"
        run_mpi_csv "$TOPO" "$RANKS" 16384 10 42 "$SWEEP_WEAK"
    done
done

# ============================================================================
echo ""
echo "=== Summaries ==="
echo "--- $SWEEP_N ---"
column -t -s, "$SWEEP_N"
echo "--- $SWEEP_N_BASE ---"
column -t -s, "$SWEEP_N_BASE"
echo "--- $SWEEP_SYNC (first 12 rows) ---"
head -13 "$SWEEP_SYNC" | column -t -s,
echo "  (... $(( $(wc -l < "$SWEEP_SYNC") - 1 )) data rows total)"
echo "--- $SWEEP_STRONG ---"
column -t -s, "$SWEEP_STRONG"
echo "--- $SWEEP_WEAK ---"
column -t -s, "$SWEEP_WEAK"
