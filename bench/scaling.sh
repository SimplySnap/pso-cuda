#!/bin/bash -l
#SBATCH --job-name=pso_scale
#SBATCH --output=bench/scaling_%j.out
#SBATCH --error=bench/scaling_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=15

# Strong + weak scaling for ring and fc.
#   Strong: fix total N_total = 4096; per-rank N = N_total / n_ranks.
#   Weak:   fix per-rank N = 1024; total N grows with ranks.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

EVAL=rastrigin
D=30
ITERS=500
SEED=42
SYNC=10
MIGRATE=5
N_TOTAL_STRONG=4096
N_PER_RANK_WEAK=1024

STRONG="bench/scaling_strong.csv"
WEAK="bench/scaling_weak.csv"
HDR="topology,evaluator,n_islands,N,D,iters,seed,eval_ms,reduce_ms,update_ms,sync_ms,total_ms,final_gbest"
echo "$HDR" > "$STRONG"
echo "$HDR" > "$WEAK"

for RANKS in 1 2 4; do
    for TOPO in ring fc; do
        N_STRONG=$(( N_TOTAL_STRONG / RANKS ))
        echo "--- strong: $TOPO ranks=$RANKS N=$N_STRONG ---"
        mpirun -np "$RANKS" "./build/pso_${TOPO}" \
            --evaluator "$EVAL" --N "$N_STRONG" --D "$D" \
            --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
            --seed "$SEED" --csv_path "$STRONG"

        echo "--- weak: $TOPO ranks=$RANKS N=$N_PER_RANK_WEAK ---"
        mpirun -np "$RANKS" "./build/pso_${TOPO}" \
            --evaluator "$EVAL" --N "$N_PER_RANK_WEAK" --D "$D" \
            --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" \
            --seed "$SEED" --csv_path "$WEAK"
    done
done

# Also capture single-GPU baselines for both N values so the analysis
# script can plot speedup against pso_cuda as well as against pso_ring -np 1.
BASE="bench/scaling_baseline.csv"
echo "impl,evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,final_gbest,achieved_bw_gbps,achieved_gflops" > "$BASE"
for N in "$N_TOTAL_STRONG" "$N_PER_RANK_WEAK"; do
    echo "--- baseline: pso_cuda N=$N ---"
    ./build/pso_cuda \
        --evaluator "$EVAL" --N "$N" --D "$D" \
        --iters "$ITERS" --seed "$SEED" --csv_path "$BASE"
done

echo ""
echo "--- $STRONG ---"
column -t -s, "$STRONG"
echo ""
echo "--- $WEAK ---"
column -t -s, "$WEAK"
echo ""
echo "--- $BASE ---"
column -t -s, "$BASE"
