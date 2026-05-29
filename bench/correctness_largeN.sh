#!/bin/bash -l
#SBATCH --job-name=pso_correct_largeN
#SBATCH --output=bench/correctness_largeN_%j.out
#SBATCH --error=bench/correctness_largeN_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=15

# Correctness sweep at the parameter regime used by §3 of M4_REPORT.md:
# sync=25, m=max(5, N/100), Rastrigin at two (D, N) cells, ranks {1,2,4,8,16}
# for ring + fc, plus single-GPU baselines and 2 Levy sanity rows.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

make -s all && make -s mpi || { echo "build FAILED"; exit 1; }

CSV=bench/correctness_largeN.csv
echo "impl,topology,n_ranks,evaluator,N,D,iters,seed,final_gbest,total_ms" > "$CSV"

ITERS=500
SEED=42
SYNC=25

migrate_for() {
    local n=$1
    local m=$(( n / 100 ))
    [ "$m" -lt 5 ] && m=5
    echo "$m"
}

run_single() {
    local eval=$1 N=$2 D=$3
    local out gbest total
    out=$(./build/pso_cuda \
        --evaluator "$eval" --N "$N" --D "$D" \
        --iters "$ITERS" --seed "$SEED" 2>&1)
    gbest=$(echo "$out" | awk -F'= ' '/^best_value/ {print $2}')
    total=$(echo "$out" | awk -F'= ' '/^total_ms/ {print $2}')
    echo "single,none,1,$eval,$N,$D,$ITERS,$SEED,$gbest,$total" >> "$CSV"
}

run_mpi() {
    local topo=$1 ranks=$2 eval=$3 N=$4 D=$5
    local m
    m=$(migrate_for "$N")
    local out gbest total
    out=$(mpirun -np "$ranks" "./build/pso_${topo}" \
        --evaluator "$eval" --N "$N" --D "$D" \
        --iters "$ITERS" --sync "$SYNC" --migrate "$m" \
        --seed "$SEED" 2>&1)
    gbest=$(echo "$out" | awk -F'= ' '/^global best_value/ {print $2}')
    total=$(echo "$out" | awk -F'= ' '/^total_ms/ {print $2}')
    echo "mpi,$topo,$ranks,$eval,$N,$D,$ITERS,$SEED,$gbest,$total" >> "$CSV"
}

# ----------------------------------------------------------------------
# Two cells: D=100 N=524288 (regime of §3.1-3.3) and D=300 N=131072 (§3.6)
# ----------------------------------------------------------------------
declare -a CELLS=("100 524288" "300 131072")

for CELL in "${CELLS[@]}"; do
    read -r D N <<< "$CELL"
    M=$(migrate_for "$N")
    echo ""
    echo "=== D=$D, N=$N, m=$M ==="

    # Single-GPU baseline:
    echo "  baseline pso_cuda"
    run_single rastrigin "$N" "$D"

    # MPI rastrigin across rank counts and topologies:
    for RANKS in 1 2 4 8 16; do
        for TOPO in ring fc; do
            echo "  rastrigin $TOPO np=$RANKS"
            run_mpi "$TOPO" "$RANKS" rastrigin "$N" "$D"
        done
    done

    # Levy sanity row (ring np=4 only):
    echo "  levy sanity (ring np=4)"
    run_mpi ring 4 levy "$N" "$D"
done

echo ""
echo "=== $CSV ==="
column -t -s, "$CSV"
