#!/bin/bash -l
#SBATCH --job-name=pso_NxRxD
#SBATCH --output=bench/sweep_NxRxD_%j.out
#SBATCH --error=bench/sweep_NxRxD_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=25

# Phase G — N x ranks x D matrix at sync=25, m = max(5, N/100).
# Each (D, ranks, topology) cell runs 2 N values bracketing the 60s ceiling.
# Per-run 90s timeout guards against runaway cells.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

# Rebuild with the new pos_local[1024] cap (supports D up to 1024).
make -s all || { echo "make all FAILED"; exit 1; }
make -s mpi || { echo "make mpi FAILED"; exit 1; }

# Quick D=300 correctness probe — Rastrigin should return finite positive value.
echo "--- D=300 correctness probe ---"
./build/pso_cuda --evaluator rastrigin --N 256 --D 300 --iters 10 --seed 42 \
    | grep -E "best_value|total_ms"

ITERS=500
SYNC=25
SEED=42

# CSVs (MPI --csv_path appends rows without headers; schema documented in
# bench/mpi_analyze.py MPI_CSV_COLS):
SWEEP=bench/sweep_NxRxD.csv
BASE=bench/sweep_NxRxD_baseline.csv
LEVY=bench/sweep_NxRxD_levy.csv
rm -f "$SWEEP" "$BASE" "$LEVY"

# Per (D, ranks) cell: 2 N values bracketing the predicted 60s ceiling.
declare -A N_GRID=(
    [30,1]="2097152 8388608"
    [30,4]="2097152 8388608"
    [30,16]="2097152 4194304"
    [100,1]="524288 2097152"
    [100,4]="524288 2097152"
    [100,16]="524288 1048576"
    [300,1]="131072 524288"
    [300,4]="131072 524288"
    [300,16]="131072 524288"
)

migrate_for() {
    local n=$1
    local m=$(( n / 100 ))
    [ "$m" -lt 5 ] && m=5
    echo "$m"
}

# ============================================================================
# Main matrix
# ============================================================================
for D in 30 100 300; do
    for RANKS in 1 4 16; do
        NS="${N_GRID[$D,$RANKS]}"
        for TOPO in ring fc; do
            first_N=$(echo "$NS" | awk '{print $1}')
            warm_m=$(migrate_for "$first_N")
            echo ""
            echo "--- warmup: D=$D ranks=$RANKS topo=$TOPO N=$first_N m=$warm_m ---"
            timeout 60s mpirun -np "$RANKS" "./build/pso_${TOPO}" \
                --evaluator rastrigin --N "$first_N" --D "$D" \
                --iters 50 --sync "$SYNC" --migrate "$warm_m" --seed "$SEED" \
                > /dev/null 2>&1 \
                || echo "  warmup skipped/timed out"

            for N in $NS; do
                MIGRATE=$(migrate_for "$N")
                echo "--- measure: D=$D ranks=$RANKS topo=$TOPO N=$N m=$MIGRATE ---"
                timeout 90s mpirun -np "$RANKS" "./build/pso_${TOPO}" \
                    --evaluator rastrigin --N "$N" --D "$D" \
                    --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
                    --seed "$SEED" --csv_path "$SWEEP" \
                    || echo "  exceeded 90s — cell skipped"
            done
        done

        # Single-GPU baseline at each (D, N) — collected once per D.
        if [ "$RANKS" = "1" ]; then
            for N in $NS; do
                echo "--- baseline: pso_cuda D=$D N=$N ---"
                timeout 90s ./build/pso_cuda \
                    --evaluator rastrigin --N "$N" --D "$D" \
                    --iters "$ITERS" --seed "$SEED" --csv_path "$BASE" \
                    || echo "  exceeded 90s — baseline skipped"
            done
        fi
    done
done

# ============================================================================
# Levy sanity rows — one per D, at np=4, ring, N=524288, m=max(5, N/100)=5242
# ============================================================================
echo ""
echo "--- Levy sanity ---"
LEVY_M=$(migrate_for 524288)
for D in 30 100 300; do
    echo "  levy: D=$D"
    timeout 90s mpirun -np 4 ./build/pso_ring \
        --evaluator levy --N 524288 --D "$D" \
        --iters "$ITERS" --sync "$SYNC" --migrate "$LEVY_M" --seed "$SEED" \
        --csv_path "$LEVY"
done

# ============================================================================
echo ""
echo "=== summaries ==="
echo "--- $SWEEP (rows: $(wc -l < "$SWEEP" 2>/dev/null || echo 0)) ---"
[ -s "$SWEEP" ] && column -t -s, "$SWEEP"
echo ""
echo "--- $BASE (rows: $(wc -l < "$BASE" 2>/dev/null || echo 0)) ---"
[ -s "$BASE" ] && column -t -s, "$BASE"
echo ""
echo "--- $LEVY (rows: $(wc -l < "$LEVY" 2>/dev/null || echo 0)) ---"
[ -s "$LEVY" ] && column -t -s, "$LEVY"
