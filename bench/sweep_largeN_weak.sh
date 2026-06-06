#!/bin/bash -l
#SBATCH --job-name=pso_weak
#SBATCH --output=bench/sweep_largeN_weak_%j.out
#SBATCH --error=bench/sweep_largeN_weak_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=30

# Phase H2 — Weak scaling at per-rank N = 8M, D = 100.
# Total particles = ranks * 8M. No timeout cap.
# Warning: np=16 cell does an Allgather of ~1.6 GB/rank/sync; could take 5+ min.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

make -s all && make -s mpi || { echo "make failed"; exit 1; }

D=100
ITERS=500
SYNC=25
SEED=42
N_PER_RANK=8388608
MIGRATE=$(( N_PER_RANK / 100 ))   # = 83886

SWEEP=bench/sweep_largeN_weak.csv
rm -f "$SWEEP"

for RANKS in 1 2 4 8 16; do
    for TOPO in ring fc; do
        echo "--- weak: $TOPO ranks=$RANKS per-rank-N=$N_PER_RANK m=$MIGRATE ---"
        mpirun -np "$RANKS" "./build/pso_${TOPO}" \
            --evaluator rastrigin --N "$N_PER_RANK" --D "$D" \
            --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
            --seed "$SEED" --csv_path "$SWEEP" \
            || echo "  (cell errored or hit slurm walltime)"
    done
done

echo ""
echo "=== $SWEEP ($(wc -l < "$SWEEP" 2>/dev/null || echo 0) rows) ==="
[ -s "$SWEEP" ] && column -t -s, "$SWEEP"
