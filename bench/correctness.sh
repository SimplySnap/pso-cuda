#!/bin/bash -l
#SBATCH --job-name=pso_correct
#SBATCH --output=bench/correctness_%j.out
#SBATCH --error=bench/correctness_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=10

# Correctness sweep across (single-GPU, ring 1/2/4, fc 1/2/4) for rastrigin and
# levy. Same seed everywhere; multi-island runs add the per-rank seed offset
# automatically inside the MPI mains.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

CSV="bench/correctness.csv"
echo "impl,topology,n_ranks,evaluator,N,D,iters,seed,final_gbest,total_ms" > "$CSV"

N=1024
D=30
ITERS=500
SEED=42
SYNC=10
MIGRATE=5

run_single() {
    local eval=$1
    local out
    out=$(./build/pso_cuda \
        --evaluator "$eval" --N "$N" --D "$D" \
        --iters "$ITERS" --seed "$SEED")
    local gbest total
    gbest=$(echo "$out" | awk -F'= ' '/^best_value/ {print $2}')
    total=$(echo "$out" | awk -F'= ' '/^total_ms/ {print $2}')
    echo "single,none,1,$eval,$N,$D,$ITERS,$SEED,$gbest,$total" >> "$CSV"
}

run_mpi() {
    local topo=$1 ranks=$2 eval=$3
    local bin="./build/pso_${topo}"
    local out
    out=$(mpirun -np "$ranks" "$bin" \
        --evaluator "$eval" --N "$N" --D "$D" \
        --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" --seed "$SEED" 2>&1)
    local gbest total
    gbest=$(echo "$out" | awk -F'= ' '/^global best_value/ {print $2}')
    total=$(echo "$out" | awk -F'= ' '/^total_ms/ {print $2}')
    echo "mpi,$topo,$ranks,$eval,$N,$D,$ITERS,$SEED,$gbest,$total" >> "$CSV"
}

for EVAL in rastrigin levy; do
    echo "--- $EVAL ---"
    run_single "$EVAL"
    for RANKS in 1 2 4; do
        run_mpi ring "$RANKS" "$EVAL"
        run_mpi fc   "$RANKS" "$EVAL"
    done
done

echo ""
echo "--- bench/correctness.csv ---"
column -t -s, "$CSV"
