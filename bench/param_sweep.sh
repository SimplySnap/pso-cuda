#!/bin/bash
# =============================================================================
# bench/param_sweep.slurm
# =============================================================================
# Sweeps c1/c2 ratio x N x {rastrigin, levy, schaffer} on a single GPU.
# Each combo is run 4 times with different seeds.
# All results are appended to pso_param_sweep.csv in the repo root.
#
# Submit from repo root:
#   sbatch bench/param_sweep.slurm
# =============================================================================

#SBATCH --job-name=pso_sweep
#SBATCH --output=logs/sweep_%j.log     # stdout+stderr — logs/ dir created below
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu-turing
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# ── environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "============================================="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "============================================="

BINARY=./build/pso_cuda
CSV=pso_param_sweep.csv

# ── sweep parameters ──────────────────────────────────────────────────────────
# Functions and their domains
#   rastrigin : D=30, bounds [-6, 6]
#   levy      : D=30, bounds [-10, 10]
#   schaffer  : D=2,  bounds [-100, 100]

declare -A FN_D=(      [rastrigin]=30  [levy]=30  [schaffer]=2   )
declare -A FN_LO=(     [rastrigin]=-6  [levy]=-10 [schaffer]=-100 )
declare -A FN_HI=(     [rastrigin]=6   [levy]=10  [schaffer]=100  )

FUNCTIONS=(rastrigin levy schaffer)
N_VALS=(256 512 1024 2048)
# c1+c2=3.0 fixed; ratio = c1 / 3.0  →  steps 0.1 .. 0.9
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
SEEDS=(1234 5678 9012 3456)
ITERS=500
W=0.7

# count total runs for progress reporting
TOTAL=$(( ${#FUNCTIONS[@]} * ${#N_VALS[@]} * ${#RATIOS[@]} * ${#SEEDS[@]} ))
DONE=0

# ── sweep loop ────────────────────────────────────────────────────────────────
for FN in "${FUNCTIONS[@]}"; do
  D=${FN_D[$FN]}
  LO=${FN_LO[$FN]}
  HI=${FN_HI[$FN]}

  for N in "${N_VALS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
      # compute c1 and c2 from ratio, keeping c1+c2=3.0
      C1=$(echo "$RATIO * 3.0" | bc -l)
      C2=$(echo "(1 - $RATIO) * 3.0" | bc -l)

      for SEED in "${SEEDS[@]}"; do
        DONE=$(( DONE + 1 ))
        echo "[$DONE/$TOTAL]  fn=$FN  N=$N  ratio=$RATIO  c1=$C1  c2=$C2  seed=$SEED"

        $BINARY                  \
          --evaluator  $FN       \
          --N          $N        \
          --D          $D        \
          --iters      $ITERS    \
          --seed       $SEED     \
          --w          $W        \
          --c1         $C1       \
          --c2         $C2       \
          --bound_lo   $LO       \
          --bound_hi   $HI       \
          --csv_path   $CSV

      done
    done
  done
done

echo "============================================="
echo "Sweep complete. Results in $CSV"
echo "End: $(date)"
echo "============================================="