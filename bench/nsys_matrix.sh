#!/bin/bash -l
#SBATCH --job-name=pso_nsys_matrix
#SBATCH --output=bench/nsys_matrix_%j.out
#SBATCH --error=bench/nsys_matrix_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=30

# Nsight matrix at D x N x ranks (ring only). 30 cells.
# Per-cell 90s timeout guards against runaway D=300 N=8M np=16 corner.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

make -s all && make -s mpi || { echo "build FAILED"; exit 1; }

# Clean any prior matrix outputs (but keep the existing summary files
# from earlier single-cell profiles: nsys_summary.txt and
# nsys_summary_largeN.txt — those have different names).
rm -f bench/trace_matrix_*.nsys-rep bench/trace_matrix_*.sqlite \
      bench/nsys_summary_D*.txt bench/nsys_run_*.out

ITERS=100
SYNC=25

migrate_for() {
    local n=$1
    local m=$(( n / 100 ))
    [ "$m" -lt 5 ] && m=5
    echo "$m"
}

# Outer ordering: D outer, N middle, ranks inner. This way if we run out of
# time, we lose the heavy D=300 high-N tail rather than missing whole rows.
for D in 30 300; do
    for N in 1024 2097152 8388608; do
        for RANKS in 1 2 4 8 16; do
            M=$(migrate_for "$N")
            TAG="D${D}_N${N}_np${RANKS}"

            echo ""
            echo "=== profile: $TAG  (m=$M) ==="

            timeout 90s mpirun -np "$RANKS" \
                nsys profile \
                    --trace=cuda,mpi \
                    --output="bench/trace_matrix_${TAG}_rank%q{OMPI_COMM_WORLD_RANK}" \
                    --force-overwrite=true \
                ./build/pso_ring \
                    --evaluator rastrigin --N "$N" --D "$D" \
                    --iters "$ITERS" --sync "$SYNC" --migrate "$M" \
                    --seed 42 \
                > "bench/nsys_run_${TAG}.out" 2>&1 \
                || { echo "  $TAG: timeout or error"; continue; }

            # Rank 0 is representative for ring with symmetric workload.
            RANK0="bench/trace_matrix_${TAG}_rank0.nsys-rep"
            if [ -f "$RANK0" ]; then
                nsys stats \
                    --report cuda_api_sum,cuda_gpu_kern_sum \
                    "$RANK0" \
                    > "bench/nsys_summary_${TAG}.txt" 2>&1
                echo "  wrote bench/nsys_summary_${TAG}.txt"
            else
                echo "  WARN: $RANK0 missing — no summary for $TAG"
            fi
        done
    done
done

echo ""
echo "=== per-cell summary file count ==="
ls bench/nsys_summary_D*.txt 2>/dev/null | wc -l
echo ""
echo "=== summary files (one per cell) ==="
ls -la bench/nsys_summary_D*.txt 2>/dev/null
