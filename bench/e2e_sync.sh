#!/bin/bash -l
#SBATCH --job-name=pso_e2e_sync
#SBATCH --output=bench/e2e_sync_%j.out
#SBATCH --error=bench/e2e_sync_%j.err
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --time=10

# =============================================================================
# End-to-end test for the on-device migration-sync gather/scatter kernels.
# -----------------------------------------------------------------------------
# The optimization replaced the per-dimension host-staged column copies in
# island_migrate_*/island_gbest_exchange with on-device gather/scatter kernels,
# so only the packed n_migrate*D migrants cross PCIe instead of whole N-float
# columns. This test guards two properties at once:
#
#   1. CORRECTNESS — same fixed seed must give a bit-identical final gbest to
#      the baseline. The change only moves data; it must not alter the result.
#   2. PERFORMANCE — the new sync_ms must beat the baseline's (that is the
#      entire reason the kernels exist).
#
# "new"      = current working tree (./build, built by `make mpi`).
# "baseline" = $BASELINE_REF (default HEAD) built in a throwaway git worktree.
#
# Run on a GPU node, e.g.:
#   sbatch bench/e2e_sync.sh
#   # or interactively:
#   salloc -p gpu-turing -N1 -n2 --ntasks-per-node=2 --gpus-per-task=1 -t10 \
#       bash -lc 'module load course/cme213/nvhpc/24.1; bash bench/e2e_sync.sh'
# =============================================================================

set -u
module load course/cme213/nvhpc/24.1
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

BASELINE_REF="${BASELINE_REF:-HEAD}"   # ref to treat as the "old" version
N=4096
D=64
ITERS=500
SYNC=10
MIGRATE=16
SEED=42
EVAL=rastrigin

fail=0
note() { printf '%s\n' "$*"; }

# --- build new (working tree) -------------------------------------------------
note ">>> building new (working tree)"
make -s mpi || { note "FAIL: make mpi (new) failed"; exit 1; }

# --- build baseline in an isolated worktree -----------------------------------
WT="$(mktemp -d "${TMPDIR:-/tmp}/pso_baseline.XXXXXX")"
cleanup() { git worktree remove --force "$WT" >/dev/null 2>&1; rm -rf "$WT"; }
trap cleanup EXIT
note ">>> building baseline ($BASELINE_REF) in $WT"
git worktree add --detach "$WT" "$BASELINE_REF" >/dev/null 2>&1 \
    || { note "FAIL: could not create worktree for $BASELINE_REF"; exit 1; }
make -s -C "$WT" mpi || { note "FAIL: make mpi (baseline) failed"; exit 1; }
cp "$WT/build/pso_ring" build/pso_ring_old
cp "$WT/build/pso_fc"   build/pso_fc_old

# --- run one config, echo "gbest sync_ms total_ms" ----------------------------
run() {
    local bin=$1
    mpirun -np 2 "$bin" \
        --evaluator "$EVAL" --N "$N" --D "$D" \
        --iters "$ITERS" --sync "$SYNC" --migrate "$MIGRATE" --seed "$SEED" 2>&1 \
    | awk -F'= ' '
        /best_value/ {g=$2}
        /^sync_ms/   {s=$2}
        /^total_ms/  {t=$2}
        END {printf "%s %s %s", g, s, t}'
}

note ""
note "config: $EVAL N=$N D=$D iters=$ITERS sync=$SYNC migrate=$MIGRATE seed=$SEED (np=2)"
printf '%-6s %-18s %-18s %-12s %-10s %s\n' \
    topo gbest_old/new sync_old/new "speedup" verdict ""

for topo in ring fc; do
    read -r g_old s_old t_old <<<"$(run ./build/pso_${topo}_old)"
    read -r g_new s_new t_new <<<"$(run ./build/pso_${topo})"

    # correctness: identical final gbest (string-equal — same seed, same math)
    corr="OK"
    if [ "$g_old" != "$g_new" ]; then corr="MISMATCH"; fail=1; fi

    # performance: new sync must be strictly faster than old
    perf="OK"
    speedup=$(awk -v a="$s_old" -v b="$s_new" 'BEGIN{ if(b>0) printf "%.2fx", a/b; else print "n/a" }')
    if awk -v a="$s_old" -v b="$s_new" 'BEGIN{exit !(b < a)}'; then :; else perf="SLOWER"; fail=1; fi

    verdict="PASS"
    [ "$corr" = OK ] && [ "$perf" = OK ] || verdict="FAIL"

    printf '%-6s %s / %s   %s / %s   %-8s [corr:%s perf:%s] %s\n' \
        "$topo" "$g_old" "$g_new" "$s_old" "$s_new" "$speedup" "$corr" "$perf" "$verdict"
done

note ""
if [ "$fail" -eq 0 ]; then
    note "E2E PASS — gbest unchanged and sync faster on every topology"
else
    note "E2E FAIL — see [corr:*]/[perf:*] above"
fi
exit "$fail"
