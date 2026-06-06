#!/bin/bash -l
#SBATCH --job-name=pso_nsys
#SBATCH --output=bench/nsys_%j.out
#SBATCH --error=bench/nsys_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=10

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"
rm -f bench/trace_ring_rank*.nsys-rep bench/trace_ring_rank*.sqlite

# Wrap nsys *inside* mpirun so each rank's process is profiled.
# %q{OMPI_COMM_WORLD_RANK} expands to the MPI rank per process.
mpirun -np 2 \
    nsys profile \
        --trace=cuda,mpi \
        --output=bench/trace_ring_rank_%q{OMPI_COMM_WORLD_RANK} \
        --force-overwrite=true \
    ./build/pso_ring \
        --evaluator rastrigin --N 1024 --D 30 \
        --iters 200 --sync 10 --migrate 5 --seed 42

# Per-rank text summaries.
for f in bench/trace_ring_rank_*.nsys-rep; do
    rank=$(basename "$f" .nsys-rep | sed 's/.*rank_//')
    echo "=== rank $rank ==="
    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,mpi_event_sum "$f"
done > bench/nsys_summary.txt 2>&1

echo "--- nsys_summary.txt ---"
cat bench/nsys_summary.txt
