#!/bin/bash
#SBATCH --job-name=pso_bench
#SBATCH --output=bench/slurm_%j.out
#SBATCH --error=bench/slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -p gpu-turing

module load cuda/12.4
module load gnu12/12.3.0
module load openmpi4/4.1.6

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"

BUILD_DIR="$REPO_DIR/build"
BENCH_DIR="$REPO_DIR/bench"
mkdir -p "$BENCH_DIR"

CSV_GPU="$BENCH_DIR/results_gpu_${SLURM_JOB_ID}.csv"
CSV_MPI="$BENCH_DIR/results_mpi_${SLURM_JOB_ID}.csv"
HIST_DIR="$BENCH_DIR/history_${SLURM_JOB_ID}"
mkdir -p "$HIST_DIR"

echo "impl,evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,final_gbest,achieved_bw_gbps,achieved_gflops" > "$CSV_GPU"
echo "topology,evaluator,n_islands,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,final_gbest" > "$CSV_MPI"

echo ">>> building binaries"
make -s all || { echo "make all FAILED — aborting"; exit 1; }
MPI_OK=1
make -s mpi  || { echo "WARNING: make mpi failed — skipping MPI runs"; MPI_OK=0; }

#schaffer excluded from MPI runs — D=30 causes dimension mismatch
EVALUATORS_GPU=("rastrigin" "levy")
EVALUATORS_MPI=("rastrigin" "levy")

N=1024
D=30
SEED=42
SYNC=10
MIGRATE=5
ITERS_LIST=(50 100 200 500)

echo "=============================="
echo "PSO benchmark — job $SLURM_JOB_ID"
echo "nodes=$SLURM_NNODES tasks=$SLURM_NTASKS"
echo "GPU CSV -> $CSV_GPU"
echo "MPI CSV -> $CSV_MPI"
echo "=============================="

# ==============================================================================
# 1. SINGLE-GPU — schaffer included here (D=30 is fine for single-GPU)
# ==============================================================================
echo ""
echo "--- single-GPU ---"
for EVAL in "${EVALUATORS_GPU[@]}"; do
    for ITERS in "${ITERS_LIST[@]}"; do
        HIST="$HIST_DIR/gpu_${EVAL}_iters${ITERS}.csv"
        echo "  gpu | $EVAL | N=$N D=$D iters=$ITERS"
        "$BUILD_DIR/pso_cuda" \
            --evaluator "$EVAL" \
            --N         "$N" \
            --D         "$D" \
            --iters     "$ITERS" \
            --seed      "$SEED" \
            --csv_path  "$CSV_GPU" \
            --history   "$HIST"
    done
done

[[ $MPI_OK -eq 0 ]] && echo "skipping MPI runs (build failed)" && exit 0

# ==============================================================================
# 2. RING topology
# ==============================================================================
echo ""
echo "--- ring topology ---"
ITERS=100
for N_ISLANDS in 2 4; do
    for EVAL in "${EVALUATORS_MPI[@]}"; do
        HIST="$HIST_DIR/ring_${EVAL}_islands${N_ISLANDS}.csv"
        echo "  ring | $EVAL | islands=$N_ISLANDS N=$N D=$D iters=$ITERS sync=$SYNC migrate=$MIGRATE"
        mpirun -np "$N_ISLANDS" "$BUILD_DIR/pso_ring" \
            --evaluator "$EVAL" \
            --N         "$N" \
            --D         "$D" \
            --iters     "$ITERS" \
            --sync      "$SYNC" \
            --migrate   "$MIGRATE" \
            --seed      "$SEED" \
            --csv_path  "$CSV_MPI" \
            --history   "$HIST"
    done
done

# ==============================================================================
# 3. FULLY-CONNECTED topology
# ==============================================================================
echo ""
echo "--- fully-connected topology ---"
for N_ISLANDS in 2 4; do
    for EVAL in "${EVALUATORS_MPI[@]}"; do
        HIST="$HIST_DIR/fc_${EVAL}_islands${N_ISLANDS}.csv"
        echo "  fc | $EVAL | islands=$N_ISLANDS N=$N D=$D iters=$ITERS sync=$SYNC migrate=$MIGRATE"
        mpirun -np "$N_ISLANDS" "$BUILD_DIR/pso_fc" \
            --evaluator "$EVAL" \
            --N         "$N" \
            --D         "$D" \
            --iters     "$ITERS" \
            --sync      "$SYNC" \
            --migrate   "$MIGRATE" \
            --seed      "$SEED" \
            --csv_path  "$CSV_MPI" \
            --history   "$HIST"
    done
done

# ==============================================================================
# 4. SYNC INTERVAL sweep — ring, rastrigin only
# ==============================================================================
echo ""
echo "--- sync interval sweep (ring, rastrigin, 4 islands) ---"
ITERS=100
N_ISLANDS=4
for SYNC_VAL in 1 5 10 25 50; do
    HIST="$HIST_DIR/ring_rastrigin_sync${SYNC_VAL}.csv"
    echo "  ring | rastrigin | sync=$SYNC_VAL"
    mpirun -np "$N_ISLANDS" "$BUILD_DIR/pso_ring" \
        --evaluator rastrigin \
        --N         "$N" \
        --D         "$D" \
        --iters     "$ITERS" \
        --sync      "$SYNC_VAL" \
        --migrate   "$MIGRATE" \
        --seed      "$SEED" \
        --csv_path  "$CSV_MPI" \
        --history   "$HIST"
done

echo ""
echo "=============================="
echo "all runs complete"
echo "GPU results -> $CSV_GPU"
echo "MPI results -> $CSV_MPI"
echo "histories   -> $HIST_DIR/"
echo "=============================="