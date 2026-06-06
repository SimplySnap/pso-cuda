#!/bin/bash -l
#SBATCH --job-name=pso_smoke
#SBATCH --output=bench/smoke_%j.out
#SBATCH --error=bench/smoke_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --time=5

module load course/cme213/nvhpc/24.1
cd "$SLURM_SUBMIT_DIR"

echo "--- pso_cuda (single GPU) ---"
./build/pso_cuda --evaluator rastrigin --N 1024 --D 30 --iters 100 --seed 42

echo ""
echo "--- pso_ring (np=2, sync=5) ---"
mpirun -np 2 ./build/pso_ring \
    --evaluator rastrigin --N 1024 --D 30 \
    --iters 50 --sync 5 --migrate 5 --seed 42

echo ""
echo "--- pso_fc (np=2, sync=5) ---"
mpirun -np 2 ./build/pso_fc \
    --evaluator rastrigin --N 1024 --D 30 \
    --iters 50 --sync 5 --migrate 5 --seed 42
