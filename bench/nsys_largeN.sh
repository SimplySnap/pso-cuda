#!/bin/bash -l
#SBATCH --job-name=pso_nsys2
#SBATCH --output=bench/nsys_largeN_%j.out
#SBATCH --error=bench/nsys_largeN_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=10

# Phase H3 — Nsight Systems profile at large N (N=2M, D=100, ring np=4, iters=100).
# Wrap nsys *inside* mpirun so each rank is profiled separately.

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

make -s all && make -s mpi || { echo "make failed"; exit 1; }

rm -f bench/trace_largeN_rank_*.nsys-rep bench/trace_largeN_rank_*.sqlite

N=2097152
D=100
ITERS=100
SYNC=25
MIGRATE=$(( N / 100 ))   # = 20971

mpirun -np 4 \
    nsys profile \
        --trace=cuda,mpi \
        --output=bench/trace_largeN_rank_%q{OMPI_COMM_WORLD_RANK} \
        --force-overwrite=true \
    ./build/pso_ring \
        --evaluator rastrigin --N "$N" --D "$D" \
        --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" --seed 42

# Per-rank stats summaries.
{
    for f in bench/trace_largeN_rank_*.nsys-rep; do
        rank=$(basename "$f" .nsys-rep | sed 's/.*rank_//')
        echo "=== rank $rank ==="
        nsys stats --report cuda_api_sum,cuda_gpu_kern_sum "$f"
    done
} > bench/nsys_summary_largeN.txt 2>&1

echo "--- nsys_summary_largeN.txt (head) ---"
head -80 bench/nsys_summary_largeN.txt
