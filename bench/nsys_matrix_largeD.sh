#!/bin/bash -l
#SBATCH --job-name=pso_nsys_largeD
#SBATCH --output=bench/nsys_matrix_largeD_%j.out
#SBATCH --error=bench/nsys_matrix_largeD_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=20

# Re-run the 5 cells that timed out in the original nsys_matrix.sh
# (D=300 × N=8M × np={1, 2, 4, 8, 16}). NO per-cell timeout this round.
# Other parameters identical to nsys_matrix.sh for table consistency.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

make -s all && make -s mpi || { echo "build FAILED"; exit 1; }

# Targeted cleanup: just the cells we are re-running. Leave the 25
# successful summary files alone.
for RANKS in 1 2 4 8 16; do
    TAG="D300_N8388608_np${RANKS}"
    rm -f bench/trace_matrix_${TAG}_rank*.nsys-rep \
          bench/trace_matrix_${TAG}_rank*.sqlite \
          bench/nsys_summary_${TAG}.txt \
          bench/nsys_run_${TAG}.out
done

ITERS=100
SYNC=25
D=300
N=8388608
M=$(( N / 100 ))

for RANKS in 1 2 4 8 16; do
    TAG="D${D}_N${N}_np${RANKS}"

    echo ""
    echo "=== profile: $TAG  (m=$M) — no timeout ==="

    mpirun -np "$RANKS" \
        nsys profile \
            --trace=cuda,mpi \
            --output="bench/trace_matrix_${TAG}_rank%q{OMPI_COMM_WORLD_RANK}" \
            --force-overwrite=true \
        ./build/pso_ring \
            --evaluator rastrigin --N "$N" --D "$D" \
            --iters "$ITERS" --sync "$SYNC" --migrate "$M" \
            --seed 42 \
        > "bench/nsys_run_${TAG}.out" 2>&1 \
        || { echo "  $TAG: ERROR"; continue; }

    RANK0="bench/trace_matrix_${TAG}_rank0.nsys-rep"
    if [ -f "$RANK0" ]; then
        nsys stats \
            --report cuda_api_sum,cuda_gpu_kern_sum \
            "$RANK0" \
            > "bench/nsys_summary_${TAG}.txt" 2>&1
        echo "  wrote bench/nsys_summary_${TAG}.txt"
    else
        echo "  WARN: $RANK0 missing for $TAG"
    fi
done

echo ""
echo "=== completed D=300 N=8M cells ==="
ls bench/nsys_summary_D300_N8388608_np*.txt 2>/dev/null
echo ""
echo "=== full matrix summary file count ==="
ls bench/nsys_summary_D*.txt 2>/dev/null | wc -l
