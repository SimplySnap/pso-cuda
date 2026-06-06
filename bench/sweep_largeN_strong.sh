#!/bin/bash -l
#SBATCH --job-name=pso_strong
#SBATCH --output=bench/sweep_largeN_strong_%j.out
#SBATCH --error=bench/sweep_largeN_strong_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=15

# Phase H1 — Strong scaling at N_total = 8M, D = 100.
# Per-rank N = N_total / ranks. No timeout cap (per user request).

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

# Use already-built binaries from prior phases; rebuild defensively in case
# someone cleaned the build dir.
make -s all && make -s mpi || { echo "make failed"; exit 1; }

D=100
ITERS=500
SYNC=25
SEED=42
N_TOTAL=8388608

SWEEP=bench/sweep_largeN_strong.csv
BASE=bench/sweep_largeN_strong_baseline.csv
rm -f "$SWEEP" "$BASE"

migrate_for() {
    local n=$1
    local m=$(( n / 100 ))
    [ "$m" -lt 5 ] && m=5
    echo "$m"
}

# Strong scaling: per-rank N shrinks as ranks grow.
for RANKS in 1 2 4 8 16; do
    N_PER=$(( N_TOTAL / RANKS ))
    MIGRATE=$(migrate_for "$N_PER")
    for TOPO in ring fc; do
        echo "--- strong: $TOPO ranks=$RANKS per-rank-N=$N_PER m=$MIGRATE ---"
        mpirun -np "$RANKS" "./build/pso_${TOPO}" \
            --evaluator rastrigin --N "$N_PER" --D "$D" \
            --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
            --seed "$SEED" --csv_path "$SWEEP" \
            || echo "  (cell errored)"
    done
done

# Single-GPU baseline at N=8M for speedup-vs-single comparison.
echo "--- baseline: pso_cuda N=$N_TOTAL D=$D ---"
./build/pso_cuda --evaluator rastrigin --N "$N_TOTAL" --D "$D" \
    --iters "$ITERS" --seed "$SEED" --csv_path "$BASE" \
    || echo "  (baseline errored)"

echo ""
echo "=== $SWEEP ($(wc -l < "$SWEEP" 2>/dev/null || echo 0) rows) ==="
[ -s "$SWEEP" ] && column -t -s, "$SWEEP"
echo ""
echo "=== $BASE ($(wc -l < "$BASE" 2>/dev/null || echo 0) rows) ==="
[ -s "$BASE" ] && column -t -s, "$BASE"
