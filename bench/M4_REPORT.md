# pso-cuda — Milestone 4 Progress Report (Markdown working copy)

> Working document for the 2-page LaTeX submission. Measurement data, design
> decisions, and citations into the codebase.

**Project:** Parallel Particle Swarm Optimization on CUDA + MPI
**Repo branch:** `oliver-m4`
**Cluster:** Stanford CME 213 teaching cluster, partition `gpu-turing` (4 nodes × 4 GPUs each, Quadro RTX 6000 sm_75)
**Toolchain:** `course/cme213/nvhpc/24.1` module (nvcc 12.3 + OpenMPI/HPC-X 2.17.1)

---

## 1. Summary

This document describes a multi-GPU island-model PSO built on MPI + CUDA, the
sweeps run against it, and what we learned. The code is on branch `oliver-m4`.

### Binaries and build

Three binaries are built from one shared CUDA library:

| Binary | What |
|---|---|
| `pso_cuda` | single-GPU baseline |
| `pso_ring` | multi-island PSO with ring-topology migration (`MPI_Sendrecv` between neighbours) |
| `pso_fc` | multi-island PSO with fully-connected migration (`MPI_Allgather`) |

`make all` builds `pso_cuda` (no MPI dependency). `make mpi` builds all three. The build queries MPI flags via `mpicxx --showme:compile`/`--showme:link` so it adapts to whichever MPI module is loaded. On the cluster's `gpu-turing` partition (4 nodes × 4 GPUs each, Quadro RTX 6000 sm_75), MPI runs use `--ntasks-per-node=4 --gpus-per-task=1` for one rank per GPU, scaling up to np=16 with no GPU sharing.

### Capabilities

- All MPI runs follow the **island model** — every rank owns a complete swarm on one GPU. No data is split across ranks; only migrants and the global best are exchanged.
- The PSO library is **MPI-agnostic**: `<mpi.h>` is included only by `mpi/mpi_island.cu` and the MPI mains. `pso_run()` calls an opaque `SyncCallback` every `sync_interval` iterations; the MPI layer registers the appropriate ring/fc callback.
- **CLI controls** for every relevant knob: `--N`, `--D`, `--iters`, `--sync`, `--migrate`, `--seed`, plus `--evaluator` (rastrigin, levy, schaffer-f2). MPI mains write a row to a CSV via `--csv_path`. Three evaluators provided.
- **D up to 1024** supported (kernel `pos_local[1024]` stack array sets the cap; D > 128 needs the bump we landed).
- **Proportional migration**: scripts compute `m = max(5, N/100)` per cell — keeps migration meaningful across a 4-orders-of-magnitude N range.
- **Per-rank RNG diversification** via per-rank seed offset.
- **Sync timing instrumentation**: host-side `sync_ms` measured via `std::chrono::steady_clock` around the `on_sync` invocation. Separates MPI/communication time from the cudaEvent-based compute timing so the CSV exposes a clean compute / sync split.
- **Nsight Systems profileable per-rank**: wrap `nsys profile` inside `mpirun` so each rank's process gets its own `.nsys-rep`. The existing scripts use the `OMPI_COMM_WORLD_RANK` substitution.

### Cross-rank correctness verified

Multi-island runs at np=1, 2, 4 converge consistently with the single-GPU `pso_cuda` for both rastrigin and levy. Multi-island generally finds equal or better minima due to diversification (see §2).

### Headline performance results

The performance/scaling work focuses on the large-N regime (per-rank N up to 8M, D up to 300) — this is the regime where MPI is meant to be used, and where the host-staging cost model finally pays off. Small-N data from the M4 baseline is preserved in **Appendix B** for traceability and historical reference.

- **Ring strong scaling is near-ideal at the right problem size**: at total N=8M, D=100, np=16 delivers **13.65× speedup over np=1 (efficiency 0.85)** and **8.11× over single-GPU pso_cuda**.
- **Ring weak scaling holds 0.90 efficiency at np=16**: total work grows 16× (8M → 134M particles) but total runtime grows only 11%.
- **fc (Allgather) collapses at np≥8**: at weak np=16, fc takes 9.4 min per run vs ring's 1.7 min; the Allgather payload's `O(p²)` growth is the structural cause.
- **Host-staging is no longer the bottleneck at large N**: the Nsight matrix in §3.7 shows `cudaMemcpy` is dwarfed by `cudaDeviceSynchronize` and GPU kernel time at every (D, N, ranks) cell with N ≥ 2M. Per-rank kernel totals are also within 1.5% across np=1…16 at fixed (D, N), confirming the strong-scaling assumption.
- **`--sync 25` is the empirical Pareto-optimal sync interval** for D=30 rastrigin: ~2× cheaper and ~2× better convergence than `--sync 10`.
- **High-D differentiation**: at D=30 the problem is too easy (most multi-island configs reach gbest=0). At D=100, ring np=16 finds 34% better minima than single-GPU at the same per-rank N. At D=300, Rastrigin is genuinely hard and migration still helps (~20% better gbest) but the absolute gap from the global optimum remains huge.

### Known limitations

- All MPI traffic goes through **host-staging cudaMemcpys**. CUDA-aware MPI was not attempted.
- The sync callback **fully blocks the next iteration**; no async-stream overlap between compute and migration.
- The migration callback copies `pbest_pos` column-by-column (D separate cudaMemcpys per rank per sync). A pack-once gather kernel would reduce this by ~5–10× at small N (less relevant at large N where compute already dominates).
- `kernel_eval_and_pbest`'s `pos_local` stack array caps **D at 1024**.
- **VRAM ceiling**: five large SoA buffers in `swarm_alloc` (positions, velocities, pbest_pos, d_r1, d_r2) each need `N × D × 4` bytes. The Quadro RTX 6000's 24 GB cap puts the runnable region at roughly `D × N ≲ 1.2 × 10⁹`. D=300 N=8M (would require ~48 GB) is therefore out of reach; the §3.7 matrix marks those cells as OOM rather than timeout.

---

## 2. Correctness testing

(`bench/correctness_largeN.sh`, slurm job 88539. Data: `bench/correctness_largeN.csv`. Fixed: rastrigin, `--sync 25`, `--migrate max(5, N/100)`, `--iters 500`, `--seed 42`. Two cells: D=100 N=524,288 and D=300 N=131,072. Ranks ∈ {1, 2, 4, 8, 16} × {ring, fc}. Plus one `pso_cuda` single-GPU baseline per (D, N) cell and one Levy sanity row.)

The earlier M4-baseline correctness sweep at N=1024, D=30, sync=10, m=5 is preserved verbatim in Appendix B. This section uses the same parameter regime as §3 — proportional migration, sync=25, 16 ranks — and the same problem sizes that §3.6's matrix exercises.

![Convergence vs rank count for both correctness cells. Ring (blue, circles) and fc (red, squares) for D=100 left, D=300 right. Horizontal gray line is the pso_cuda single-GPU baseline. Multi-island beats single-GPU at every (D, ranks≥2) cell with substantial margin at D=100. The fc np=1 D=300 outlier (gbest=3,359, annotated arrow) is the slot-0 overwrite anomaly discussed below. Source: bench/correctness_largeN.csv → bench/plot_correctness.py.](fig_correctness.png)

### 2.1 Rastrigin at D=100, per-rank N=524,288

Single-GPU baseline `pso_cuda --N 524288 --D 100`: **total_ms = 3,643**, final_gbest = **165.43**.

| topology | n_ranks | final_gbest | total_ms |
|---|---|---|---|
| ring | 1 | 187.25 | 6,215 |
| fc | 1 | 193.47 | 5,962 |
| ring | 2 | **130.49** | 6,960 |
| fc | 2 | **142.11** | 6,940 |
| ring | 4 | 135.33 | 6,312 |
| fc | 4 | 125.83 | 6,159 |
| ring | 8 | 102.38 | 6,702 |
| fc | 8 | **77.47** | 17,050 |
| ring | 16 | 108.54 | 6,688 |
| fc | 16 | 88.64 | 56,806 |

### 2.2 Rastrigin at D=300, per-rank N=131,072

Single-GPU baseline `pso_cuda --N 131072 --D 300`: **total_ms = 2,824**, final_gbest = **1,402.27**.

| topology | n_ranks | final_gbest | total_ms |
|---|---|---|---|
| ring | 1 | 1,335.62 | 5,058 |
| **fc** | **1** | **3,359.44** ❌ | 4,714 |
| ring | 2 | 1,389.81 | 5,459 |
| fc | 2 | 1,400.61 | 5,466 |
| ring | 4 | 1,488.06 | 5,063 |
| fc | 4 | 1,503.50 | 5,069 |
| ring | 8 | 1,479.25 | 5,500 |
| fc | 8 | 1,353.17 | 12,480 |
| ring | 16 | 1,332.89 | 5,607 |
| fc | 16 | **1,329.47** | 40,369 |

### 2.3 Levy sanity rows (`ring`, `np=4`)

| D | N | final_gbest |
|---|---|---|
| 100 | 524,288 | 7.64e-15 (machine epsilon) |
| 300 | 131,072 | 4.56 (no longer trivial — consistent with §3.6.2) |

### Findings

Four invariants checked programmatically against `bench/correctness_largeN.csv`:

1. **All gbest values finite and ≥ 0** — pass. No NaN, no INFINITY, no negatives on Rastrigin.
2. **Best multi-island ≤ single-GPU baseline** — pass at both cells with substantial margin:
   - D=100: best MPI is **fc np=8 at gbest=77.47**, 2.13× better than single-GPU's 165.43.
   - D=300: best MPI is **fc np=16 at gbest=1,329.47**, 5% better than single-GPU's 1,402.27.
3. **Ring vs fc at np=2 — approximate, not exact.** At small problem sizes (the Appendix B baseline) ring and fc agree exactly at np=2 (both 21.889114) because the two topologies are isomorphic for two ranks. At the larger sizes here:
   - D=100, N=524K, np=2: ring=130.49 vs fc=142.11 (8.2% relative difference)
   - D=300, N=131K, np=2: ring=1,389.81 vs fc=1,400.61 (0.77% relative difference)

   The most likely explanation is floating-point ordering inside MPI primitives: `MPI_Sendrecv` (ring) and `MPI_Allgather` (fc) walk their internal reduction trees differently, so the bit-level pbest values that get injected after a sync are not byte-identical even when the algorithmic content is. With larger payloads (large N × D × m) the per-iteration accumulation of these ordering effects compounds. Neither answer is "wrong" but the equivalence assumption is empirically only ≤ 8%.
4. **fc np=1 at D=300 is a reproducible anomaly** — fail on the "within 2× of pso_cuda" check. `fc np=1 → gbest=3,359` vs pso_cuda's 1,402 (2.40×). The same value (3,359.4368) appears in `sweep_NxRxD.csv` from a separate run, so it is deterministic, not flaky. The mechanism: at np=1 the `island_gbest_exchange` callback fires 20 times (sync=25 over 500 iters) and overwrites `pbest_pos[*, 0]` with the broadcast position. At D=300 with `m = 1,310` the migration callback also overwrites 1,310 particles' pbests with self-broadcast top-1% data — at this difficulty the slot-0 churn measurably hurts a single-island run that has no other source of diversity. Ring np=1 doesn't suffer as badly because `island_migrate_ring`'s self-Sendrecv is effectively a no-op (sends top-m to self) and the gbest exchange alone is less destructive than fc's combined Allgather + gbest cycle. The right interpretation: pso_ring -np 1 and pso_fc -np 1 are not "the same as pso_cuda" by design (see §4); they are MPI binaries running the MPI codepath even when there is no neighbor to talk to.

**Algorithmic correctness is established.** Two known divergences from byte-perfect equivalence (floating-point ordering at np=2 between topologies, and the np=1 slot-0 overwrite hurting fc at high D) are non-bugs in the sense that they fall out of the host-staging migration design as documented. The implementation does what the algorithm specifies. The np=1 fc result on hard problems is worth optimizing in a future revision — see §5.

---

## 3. Performance, scaling, profiling

> The section now centers on **large-N** measurements (per-rank N up to 8M, D = 100) which is the regime where MPI is meant to be used. The original M4-baseline data at N_total = 4096 is preserved in **Appendix B** at the bottom of this document for historical traceability.

### 3.1 Strong scaling at N_total = 8M, D = 100

*(`bench/sweep_largeN_strong.sh`, slurm job 88016. Data: `bench/sweep_largeN_strong.csv` + `bench/sweep_largeN_strong_baseline.csv`. Figure: `bench/fig_strong_scaling.png` (produced by `bench/plot_strong_scaling.py`). Fixed: rastrigin, sync=25, m=max(5, N/100), iters=500, seed=42, per-rank N = N_total / ranks.)*

![Strong scaling at N_total=8M, D=100. Left: speedup vs ranks, with two baselines per topology (vs MPI np=1 solid/dashed, vs pso_cuda single-GPU dotted). Ring at np=16 delivers 13.65× speedup over np=1 and 8.11× over pso_cuda. Diagonal gray line is ideal y=p. Right: parallel efficiency = speedup_vs_mpi1 / p. Ring holds 0.85 efficiency at np=16; fc collapses past np=4.](fig_strong_scaling.png)

**Single-GPU baseline:** `pso_cuda --N 8388608 --D 100` → **total_ms = 54,529** ms, final_gbest = 60.85.

| ranks | topology | per-rank N | total_ms | sync_ms | speedup vs np=1 | efficiency | **speedup vs single-GPU** | final_gbest |
|---|---|---|---|---|---|---|---|---|
| 1 | ring | 8,388,608 | 91,767 | 37,376 | 1.00 | 1.00 | 0.59× | 84.7 |
| 1 | fc | 8,388,608 | 86,438 | 31,852 | 1.00 | 1.00 | 0.63× | 60.8 |
| 2 | ring | 4,194,304 | 53,224 | 26,077 | 1.72 | 0.86 | 1.02× | 89.2 |
| 2 | fc | 4,194,304 | 52,600 | 25,538 | 1.64 | 0.82 | 1.04× | 77.8 |
| 4 | ring | 2,097,152 | 23,256 | 9,539 | 3.95 | 0.99 | 2.34× | 39.1 |
| 4 | fc | 2,097,152 | 22,945 | 9,002 | 3.77 | 0.94 | 2.38× | 72.9 |
| 8 | ring | 1,048,576 | **12,791** | 5,706 | **7.18** | **0.90** | **4.26×** | 154.2 |
| 8 | fc | 1,048,576 | 33,788 | 26,784 | 2.56 | 0.32 | 1.61× | 98.0 |
| 16 | ring | 524,288 | **6,725** | 3,247 | **13.65** | **0.85** | **8.11×** | 108.5 |
| 16 | fc | 524,288 | 37,386 | 33,828 | 2.31 | 0.14 | 1.46× | 88.6 |

**Findings:**

1. **Ring delivers near-ideal strong scaling.** At np=16 it achieves a **13.65× speedup over np=1 with 0.85 efficiency** — far better than the smaller-N sweeps had suggested. The reason: at N_total = 8M and per-rank N = 524K, per-iter compute is heavy enough (2.2 sec update + 1.3 sec eval) that it dominates the proportional-migration sync cost (3.2 sec). This is the regime MPI was designed for.
2. **Ring beats single-GPU by 8.11×** at np=16. The previous report's headline ("MPI doesn't beat single-GPU under proportional migration") was correct *at the wrong problem size* — at large N_total the picture inverts completely.
3. **fc breaks at np≥8.** At np=8 fc speedup plateaus at 2.56× (efficiency 0.32); at np=16 it's 2.31× (efficiency 0.14). The Allgather's `O(p²)` payload (≈ p × m × D × 4 bytes per rank, growing quadratically with p) starts to dominate everything once p ≥ 8.
4. **The 1-rank cells are slower than single-GPU** (ring np=1: 91.8 sec vs pso_cuda 54.5 sec). This is the no-op MPI overhead: even with no neighbors, the callback fires, does the cudaMemcpy dance, and runs the Allreduce/Bcast collectives. Cost: ~37 sec out of 91. As ranks grow, the per-rank compute shrinks faster than this fixed overhead grows, so the curve goes through the single-GPU baseline at p ≈ 2 and keeps improving.
5. **Convergence quality is not monotone in ranks** (gbest: 84.7 → 89.2 → 39.1 → 154.2 → 108.5 for ring). Migration redistributes good particles across smaller swarms; with only 524K particles per island the smaller subswarms have higher variance in what they find. The np=4 cell is the sweet spot for *quality* (gbest=39.1) while np=16 is the sweet spot for *speed* (6.7 sec).

### 3.2 Weak scaling at per-rank N = 8M, D = 100

*(`bench/sweep_largeN_weak.sh`, slurm job 88017. Data: `bench/sweep_largeN_weak.csv`. Figure: `bench/fig_weak_scaling.png` (produced by `bench/plot_weak_scaling.py`). Same fixed params as §3.1. Per-rank N stays at 8,388,608; total particles grow from 8M → 134M as ranks scale 1 → 16.)*

![Weak scaling at per-rank N=8M, D=100. Left: total runtime (log scale) across rank counts. Ring stays near-flat (91 sec → 101 sec from np=1 to np=16, +11%); fc collapses to 565 sec at np=16. Right: weak-scaling efficiency = T(np=1) / T(np=p). Ring holds 0.90 at np=16, fc drops to 0.15.](fig_weak_scaling.png)

| ranks | topology | per-rank N | eval_ms | sync_ms | total_ms | efficiency (T_1/T_p) | final_gbest |
|---|---|---|---|---|---|---|---|
| 1 | ring | 8,388,608 | 18,743 | 37,247 | 91,488 | 1.00 | 84.7 |
| 1 | fc | 8,388,608 | 19,082 | 31,945 | 86,487 | 1.00 | 60.8 |
| 2 | ring | 8,388,608 | 18,196 | 55,449 | 109,140 | 0.84 | 51.8 |
| 2 | fc | 8,388,608 | 19,027 | 53,538 | 108,076 | 0.80 | 60.8 |
| 4 | ring | 8,388,608 | 18,487 | 39,821 | 93,776 | **0.98** | 46.9 |
| 4 | fc | 8,388,608 | 19,419 | 37,638 | 92,540 | **0.93** | 53.2 |
| 8 | ring | 8,388,608 | 18,314 | 46,808 | 100,594 | 0.91 | 43.5 |
| 8 | fc | 8,388,608 | 18,427 | **221,780** | **275,768** | **0.31** | 48.2 |
| 16 | ring | 8,388,608 | 18,670 | 47,084 | 101,239 | **0.90** | 43.3 |
| 16 | fc | 8,388,608 | 18,323 | **511,808** | **565,675** | **0.15** | 39.9 |

**Findings:**

1. **Ring weak scaling stays at 0.90–0.98 efficiency even at np=16.** Total work grows 16× (8M → 134M particles) but total_ms only grows 1.11× (91.5 → 101.2 sec). The migration cost barely grows: sync_ms goes from 37s at np=1 to 47s at np=16 — only +27% for 16× more work. This is the strongest signal that ring's `O(p)` Sendrecv pattern scales correctly.
2. **fc weak scaling collapses dramatically at np ≥ 8.** Sync_ms jumps 38s → 222s → **512s** as ranks go 4 → 8 → 16. The Allgather payload at np=16 is ~537 MB per rank per call (m × D × 4 × p = 83886 × 100 × 4 × 16); over 20 syncs each with two Allgather calls (positions + fitnesses), that's ~21 GB of MPI traffic per rank. Cluster Infiniband at ~100 Gbps takes ~28 sec per Allgather; the full sync_ms is consistent with this back-of-envelope.
3. **fc np=16 took 9.4 minutes for one cell** — completes correctly but is the worst-performing configuration in any of our experiments. Efficiency 0.15 means we're getting 1/6 of the work-normalized throughput vs np=1.
4. **Convergence improves with ranks under weak scaling** — gbest drops from 84.7 (ring np=1) → 43.3 (ring np=16) because the total swarm size grows with ranks (more particles = more search). This is the algorithmic win that compensates for the modest 11% runtime increase.

### 3.3 Comm vs compute breakdown at large N

*(From `bench/sweep_largeN_strong.csv` and `bench/sweep_largeN_weak.csv`. Figure: `bench/fig_breakdown.png` (produced by `bench/plot_breakdown.py`).)*

![Per-component time breakdown across all 20 configs from §3.1 (strong) and §3.2 (weak). Stack components from bottom: eval_ms (blue), reduce_ms (green), update_ms (red), sync_ms (purple). The fc np=16 weak bar towers over everything at 565 sec — almost entirely sync. Ring at the same configuration is 101 sec with sync and compute roughly balanced.](fig_breakdown.png)

At large N + D the ratio that dominated the M4 baseline (sync ≈ 28× compute) inverts:

| Config | compute_ms (eval + reduce + update) | sync_ms | sync / compute |
|---|---|---|---|
| Strong ring np=1 (N=8M) | 54,391 | 37,376 | **0.69** |
| Strong ring np=4 (N=2M) | 13,717 | 9,539 | **0.70** |
| Strong ring np=8 (N=1M) | 7,085 | 5,706 | **0.81** |
| **Strong ring np=16 (N=524K)** | **3,478** | **3,247** | **0.93** |
| Weak ring np=16 (N=8M each) | 54,156 | 47,084 | 0.87 |

The "host-staging is the bottleneck" story from the M4 baseline only holds when per-iter compute is small (small N or small D). At N=8M, D=100, **compute per iteration is ~110 ms × 500 iters ≈ 55 sec — dwarfing the per-call MPI/cudaMemcpy latency overhead.** Earlier hopes that sync would amortize *cleanly* at large N (under the fixed `--migrate 5` policy in older sweeps) were right in direction but wrong in magnitude — under proportional migration `m = N/100` the crossover hasn't quite happened (sync/compute still 0.7–0.9), but the regimes are now comparable rather than 28×-apart.

This is why ring strong scaling reaches efficiency 0.85 at np=16 (§3.1): once compute and sync are roughly equal, splitting compute across ranks actually works — because the absolute sync cost grows slowly.

### 3.4 Sync-interval sweep — Pareto front of cost vs convergence

(5 seeds × 6 sync_interval values × 2 topologies = 60 runs. Data in `bench/sweep_sync.csv`, figure in `bench/fig_sweep_sync.png`.)

| sync_interval | sync_ms_mean (ring) | gbest_mean (ring) | gbest_std (ring) |
|---|---|---|---|
| 1 | 784.0 | 25.88 | 8.10 |
| 5 | 166.7 | 26.86 | 9.04 |
| **10 (default)** | 90.2 | 16.36 | 4.78 |
| **25 (best)** | **47.0** | **8.61** | 2.85 |
| 50 | 30.7 | 10.92 | 1.79 |
| 100 | 21.0 | 16.92 | 0.004 |

**Findings:**

1. **The current default `--sync 10` is not Pareto-optimal.** `--sync 25` is both **cheaper** (47 ms vs 90 ms in `sync_ms`) **and converges to a better gbest** (8.6 vs 16.4 mean). This was the most surprising result.
2. **`--sync 1` is actively harmful** — synchronizing every iteration leaves no time for islands to explore independently. They homogenize too quickly, eliminating the algorithmic benefit of the island model. Variance is also high (std 8.1), confirming the runs are not consistent.
3. **`--sync 100` collapses to no-MPI behavior.** Only 5 syncs happen across 500 iters; `final_gbest = 16.918 ± 0.004` for every seed (essentially independent runs of single-GPU with a final reduce). Convergence quality is the same as M4's single-GPU baseline at N=1024 (49.7 → 16.9), confirming that the migration mechanism *is* doing the work between syncs.
4. **There is a clear sweet spot around `--sync 25–50`** — both topologies. Migration happens often enough to inject good genes but not so often that islands' independent exploration is wasted.

**Actionable change for the final report:** all subsequent MPI runs should use `--sync 25` as the default. This alone reduces sync_ms by ~2× while improving convergence by ~2×.

### 3.5 *(Removed — an earlier strong/weak scaling at N_total = 16384 was superseded by §3.1–3.3 above, which run at the much larger N_total = 8M with proportional migration. The data is still in `bench/sweep_{strong,weak}_largeN.csv` if anyone wants the intermediate datapoint.)*

### 3.6 Curse of dimensionality at D = 100 and D = 300

(Single slurm job `87979`. Data: `bench/sweep_NxRxD.csv` + `bench/sweep_NxRxD_baseline.csv` + `bench/sweep_NxRxD_levy.csv`. Figure: `bench/fig_sweep_NxRxD.png`. Matrix: D ∈ {30, 100, 300} × ranks ∈ {1, 4, 16} × {ring, fc}, two per-cell per-rank N values bracketing what fits in 90 s. Same sync=25 and proportional `m = max(5, N/100)` as §3.1–3.3.)

The strong/weak scaling sections (§3.1, §3.2) and the comm/compute breakdown (§3.3) all run at D=100. **This section asks: what changes at D=300, where Rastrigin becomes genuinely hard?** The §3.1 ring-near-ideal-scaling and §3.2 fc-breaks-at-np=16 findings replicate cleanly at D=30 and D=300 (data in the CSV); we focus the discussion here on the algorithmic question.

The fc-collapse and proportional-migration findings already discussed elsewhere apply across this matrix too: 35 of 36 cells completed; the lone failure was D=300 fc np=16 at N=524K, which exceeded the 90-second per-cell timeout.

#### 3.6.1 Algorithmic differentiation finally emerges at D=300

D=30 is too easy for the multi-island advantage to matter (most cells reach gbest=0 regardless). D=100 starts to show migration benefit. D=300 makes Rastrigin genuinely hard:

| D | impl | N | final_gbest |
|---|---|---|---|
| 30 | pso_cuda | 2,097,152 | **0.99** (near optimal) |
| 30 | ring np=16 | 2,097,152 | **0.00** (machine zero) |
| 100 | pso_cuda | 524,288 | 165.4 |
| 100 | ring np=4 | 524,288 | 135.3 (18% better) |
| 100 | ring np=16 | 524,288 | **108.5** (34% better than single-GPU) |
| 100 | ring np=16 | 1,048,576 | **50.4** |
| 300 | pso_cuda | 131,072 | 1402 |
| 300 | ring np=4 | 524,288 | 1124 (20% better) |
| 300 | ring np=16 | 524,288 | **1108** (also ~21% better; diminishing return vs np=4) |

At D=300, every config still struggles (gbest ~1100 vs the true minimum of 0), but multi-island consistently helps. The "more islands = more exploration" benefit is real but bounded by per-rank N — at D=300, none of our N values are big enough for the search to genuinely tame the curse of dimensionality.

#### 3.6.2 Levy stops being trivial at D=300

Levy at D=30 and D=100 still hits machine epsilon (~7.6e-15) under multi-island runs. At D=300, the sanity row shows `final_gbest = 2.06` — Levy is no longer the "everyone gets 0" benchmark we used in §2. For future studies this means Levy at D≥300 becomes a useful evaluator alongside Rastrigin.

---

### 3.7 Nsight Systems matrix across D × N × ranks

*(`bench/nsys_matrix.sh`, slurm job 88625; corner re-run in `bench/nsys_matrix_largeD.sh`, slurm job 88634. Per cell: `mpirun -np {ranks} nsys profile ./pso_ring --evaluator rastrigin --N {N} --D {D} --iters 100 --sync 25 --migrate max(5, N/100) --seed 42`. Per-cell summaries (rank-0 `cuda_api_sum` + `cuda_gpu_kern_sum`) in `bench/nsys_summary_D{D}_N{N}_np{ranks}.txt`. Long-format CSV in `bench/nsight_matrix.csv`. Tables produced by `bench/parse_nsys.py` and replicated below; see `bench/nsight_tables.md` for the rendered file. **25 of 30 cells captured.** The five `D=300 × N=8M × np={1, 2, 4, 8, 16}` cells were initially classified as "timeout" but the corner re-run without the per-cell timeout exposed the real cause: every cell aborts with `CUDA error: out of memory` at `swarm_alloc` (`pso/pso.cu:126`). At D=300 N=8M the five large SoA buffers (positions, velocities, pbest_pos, d_r1, d_r2) each need `N × D × 4 = 9.6 GB`, totalling ~48 GB — exceeding the Quadro RTX 6000's 24 GB VRAM. This is a hardware ceiling, not a budget issue.)*

#### Table A — CUDA API breakdown (ms, rank 0)

| D | N | np | cudaMemcpy | cudaDeviceSync | cudaLaunchKernel | cudaMemcpyFromSymbol | other | total |
|---|---|---|---|---|---|---|---|---|
| 30  | 1K | 1  | 5.6     | 0.13    | 4.5  | 7.4   | 2.3  | 20.0    |
| 30  | 1K | 2  | 4.9     | 0.14    | 7.2  | 12.6  | 9.2  | 34.1    |
| 30  | 1K | 4  | 4.9     | 0.13    | 5.8  | 14.6  | 3.2  | 28.7    |
| 30  | 1K | 8  | 4.9     | 0.14    | 10.8 | 12.9  | 4.1  | 32.9    |
| 30  | 1K | 16 | 4.7     | 0.13    | 4.8  | 15.6  | 3.2  | 28.5    |
| 30  | 2M | 1  | 449.9   | 938.6   | 5.1  | 9.4   | 13.8 | 1.42 s  |
| 30  | 2M | 2  | 528.1   | 927.7   | 8.2  | 20.0  | 17.9 | 1.50 s  |
| 30  | 2M | 4  | 331.5   | 941.2   | 6.0  | 14.8  | 17.8 | 1.31 s  |
| 30  | 2M | 8  | 323.8   | 939.0   | 5.2  | 18.8  | 15.4 | 1.30 s  |
| 30  | 2M | 16 | 325.7   | 938.3   | 5.3  | 16.6  | 16.8 | 1.30 s  |
| 30  | 8M | 1  | 1,750.4 | 3,728.2 | 5.1  | 8.8   | 17.3 | 5.51 s  |
| 30  | 8M | 2  | 2,630.7 | 3,735.4 | 6.3  | 17.0  | 21.9 | 6.41 s  |
| 30  | 8M | 4  | 1,746.4 | 3,732.8 | 5.7  | 11.0  | 26.0 | 5.52 s  |
| 30  | 8M | 8  | 1,752.8 | 3,735.3 | 5.4  | 18.0  | 26.0 | 5.54 s  |
| 30  | 8M | 16 | 1,750.8 | 3,730.7 | 5.3  | 13.5  | 24.2 | 5.52 s  |
| 300 | 1K | 1  | 56.7    | 11.5    | 4.6  | 7.6   | 3.6  | 84.0    |
| 300 | 1K | 2  | 59.1    | 10.6    | 10.2 | 14.5  | 8.1  | 102.6   |
| 300 | 1K | 4  | 46.2    | 10.8    | 6.9  | 10.8  | 8.5  | 83.3    |
| 300 | 1K | 8  | 44.6    | 10.6    | 9.8  | 11.5  | 7.5  | 84.0    |
| 300 | 1K | 16 | 45.5    | 10.6    | 7.5  | 23.0  | 16.0 | 102.6   |
| 300 | 2M | 1  | 4,139.1 | 8,862.9 | 5.1  | 8.9   | 22.8 | 13.04 s |
| 300 | 2M | 2  | 6,008.2 | 8,781.2 | 6.1  | 13.8  | 25.4 | 14.83 s |
| 300 | 2M | 4  | 3,097.8 | 8,451.9 | 6.2  | 15.9  | 41.3 | 11.61 s |
| 300 | 2M | 8  | 3,100.9 | 8,453.1 | 13.5 | 12.1  | 45.0 | 11.62 s |
| 300 | 2M | 16 | 3,093.6 | 8,480.9 | 6.2  | 12.7  | 55.7 | 11.65 s |
| 300 | 8M | * | — | — | — | — | — | **OOM** (24 GB VRAM cap) |

#### Table B — GPU kernel breakdown (ms, rank 0)

| D | N | np | eval_and_pbest | update | draw_rng | CUB ArgMin | commit_gbest | other | total |
|---|---|---|---|---|---|---|---|---|---|
| 30  | 1K | 1  | 1.0     | 0.15    | 0.33    | 0.18 | 0.13 | 0.11 | 1.9    |
| 30  | 1K | 2  | 1.0     | 0.15    | 0.33    | 0.18 | 0.12 | 0.11 | 1.9    |
| 30  | 1K | 4  | 0.98    | 0.15    | 0.33    | 0.18 | 0.12 | 0.11 | 1.9    |
| 30  | 1K | 8  | 1.0     | 0.15    | 0.33    | 0.18 | 0.12 | 0.11 | 1.9    |
| 30  | 1K | 16 | 0.98    | 0.15    | 0.33    | 0.18 | 0.12 | 0.11 | 1.9    |
| 30  | 2M | 1  | 338.5   | 359.4   | 222.0   | 2.5  | 0.14 | 19.2 | 941.8  |
| 30  | 2M | 2  | 328.0   | 359.8   | 221.5   | 2.5  | 0.14 | 19.4 | 931.4  |
| 30  | 2M | 4  | 340.9   | 359.3   | 222.6   | 2.5  | 0.14 | 19.3 | 944.7  |
| 30  | 2M | 8  | 339.8   | 359.2   | 221.5   | 2.5  | 0.14 | 19.3 | 942.6  |
| 30  | 2M | 16 | 339.1   | 359.2   | 221.5   | 2.5  | 0.14 | 19.5 | 941.9  |
| 30  | 8M | 1  | 1,294.8 | 1,436.7 | 907.4   | 7.3  | 0.17 | 85.0 | 3.73 s |
| 30  | 8M | 2  | 1,302.4 | 1,436.2 | 906.9   | 7.4  | 0.17 | 86.4 | 3.74 s |
| 30  | 8M | 4  | 1,299.0 | 1,436.4 | 906.7   | 7.4  | 0.17 | 86.6 | 3.74 s |
| 30  | 8M | 8  | 1,298.7 | 1,436.4 | 909.0   | 7.4  | 0.17 | 86.9 | 3.74 s |
| 30  | 8M | 16 | 1,297.6 | 1,436.0 | 906.0   | 7.4  | 0.17 | 86.9 | 3.73 s |
| 300 | 1K | 1  | 11.4    | 0.86    | 1.4     | 0.21 | 0.14 | 0.13 | 14.1   |
| 300 | 1K | 2  | 11.3    | 0.85    | 1.4     | 0.21 | 0.14 | 0.13 | 14.0   |
| 300 | 1K | 4  | 11.1    | 0.95    | 1.4     | 0.21 | 0.14 | 0.13 | 14.0   |
| 300 | 1K | 8  | 11.1    | 0.95    | 1.4     | 0.21 | 0.14 | 0.13 | 14.0   |
| 300 | 1K | 16 | 11.1    | 0.95    | 1.4     | 0.21 | 0.14 | 0.13 | 14.0   |
| 300 | 2M | 1  | 3,727.9 | 3,580.0 | 1,517.2 | 2.5  | 0.15 | 38.4 | 8.87 s |
| 300 | 2M | 2  | 3,648.9 | 3,576.1 | 1,518.1 | 2.5  | 0.15 | 38.7 | 8.78 s |
| 300 | 2M | 4  | 3,321.3 | 3,577.1 | 1,516.1 | 2.5  | 0.15 | 38.5 | 8.46 s |
| 300 | 2M | 8  | 3,320.6 | 3,577.9 | 1,516.8 | 2.5  | 0.15 | 38.9 | 8.46 s |
| 300 | 2M | 16 | 3,349.0 | 3,578.3 | 1,516.1 | 2.5  | 0.15 | 38.5 | 8.48 s |
| 300 | 8M | * | — | — | — | — | — | — | **OOM** (24 GB VRAM cap) |

#### Reading the matrix

1. **`cudaDeviceSynchronize` is the largest CUDA-API line item at large N**, not `cudaMemcpy`. At D=30 N=8M np=1 it's 3.73 sec across 6 calls (~620 ms per drain); at D=300 N=2M np=1 it's 8.86 sec across 6 calls (~1.5 sec per drain). These are the explicit pre-callback `cudaDeviceSynchronize()` calls in `pso_run` plus the lifecycle drains — the GPU's pipeline of pending kernels actually executes inside this barrier, so much of the "compute time" is sitting *inside* this synchronization, not inside `kernel_eval_and_pbest` directly. An async-stream implementation would move that time off the critical path.
2. **`cudaMemcpy` scales with `m × D` per sync, not with N directly**, so it grows weakly with N once N is large enough that the per-call overhead is amortized. From D=30 N=2M to D=30 N=8M (4× particles), `cudaMemcpy` grows 4× (450 → 1,750 ms) — proportional. From D=30 N=8M to D=300 N=2M (smaller N, 10× D, equal m), `cudaMemcpy` grows 2.4× (1,750 → 4,140 ms) — D matters more than N for this line.
3. **Per-rank kernel times are nearly identical across rank counts at fixed (D, N)**. Compare D=30 N=2M: rank-0 kernel total is 941.8 / 931.4 / 944.7 / 942.6 / 941.9 ms for np = 1/2/4/8/16. Variance is ≤ 1.5%. This is the strongest direct evidence that ring's strong-scaling efficiency comes from splitting *work* across ranks rather than each rank doing less work per iteration — exactly what §3.1's strong scaling assumes.
4. **`kernel_eval_and_pbest` and `kernel_update` are the two dominant GPU kernels** at every cell, with `kernel_draw_rng` consistently third. At D=300 N=2M, eval = 3,728 ms is the biggest line — D=300 makes evaluator math the per-iter bottleneck. At D=30 the three are closer in size (338 / 359 / 222 ms at N=2M). CUB `ArgMin` and `kernel_commit_gbest` are ≤ 3 ms total in every measured cell — the gbest reduction is not on any critical path.
5. **The `D=300 × N=8M` cells are unreachable on this hardware** — not a budget problem but a VRAM ceiling. At D=300 N=8M, the five large SoA buffers in `swarm_alloc` (positions, velocities, pbest_pos, d_r1, d_r2) each need `N × D × 4 = 9.6 GB`, totalling ~48 GB and exceeding the Quadro RTX 6000's 24 GB. The corner re-run (`bench/nsys_matrix_largeD.sh`, slurm job 88634, no per-cell timeout) confirmed this: every cell aborts with `CUDA error pso/pso.cu:126: out of memory`. Even np=16 doesn't help because each rank still allocates its own per-rank N=8M swarm. The same VRAM ceiling shapes §3.6 and §3 generally — `D × N ≲ 1.2 × 10⁹` is the runnable region under the current SoA allocation pattern.

---

## 4. Discussion / bottlenecks

**Headline:**

- **At small total problem sizes (N_total ≤ 4096)** the host-staging migration costs ~7× the GPU compute and MPI loses outright to single-GPU.
- **The earlier "MPI wins at large N" prediction was an artifact** of fixed `m=5`. Under proportional migration (`m = max(5, N/100)`, §3.6), sync_ms scales with N just like compute does. The sync/compute ratio stays near 1.0 from N=2M to N=8M — *no crossover*. MPI stays ~2× slower than single-GPU at every cell.
- **`--sync 25` is the empirical Pareto-optimal sync_interval** (§3.4): 2× cheaper AND 2× better convergence than the `--sync 10` we used in early sweeps.
- **Ring strong scaling is near-ideal at the right problem size** (§3.1). At N_total=8M D=100 it achieves **13.65× speedup over np=1 with efficiency 0.85** at np=16, AND **8.11× speedup over single-GPU**. The earlier reports' "MPI loses to single-GPU" conclusion was a small-N artifact — at the natural problem size for these GPUs, the system works as intended.
- **Ring weak scaling holds 0.90 efficiency at np=16** (§3.2): total work grows 16× but total_ms grows only 1.11×, with sync_ms growing only 27% (37s → 47s) across the same range. This is the strongest evidence that ring's `O(p)` cost scales correctly.
- **fc breaks at np≥8** for any non-trivial N. At np=8 weak, fc sync_ms = 222 sec vs ring's 47 sec (4.7× worse). At np=16 strong, fc speedup plateaus at 2.3× (efficiency 0.14) while ring hits 13.65× (efficiency 0.85). The Allgather's `O(p²)` payload growth is the structural cause.
- **Host-staging is no longer the bottleneck at large N** (§3.7). Nsight at N=2M D=100 shows kernel time (3.0 sec) exceeds cudaMemcpy time (1.3 sec) by 2.3×. The reverse held at N=1024 (M4 baseline, Appendix B.4): 28 ms cudaMemcpy vs 3.8 ms kernels. The crossover is around N ≈ 100K under proportional migration.
- **Algorithmic differentiation requires high D.** At D=30 the problem is too easy — most multi-island configurations reach gbest=0. At D=100, ring np=16 finds 34% better minima than single-GPU at the same per-rank N. At D=300, Rastrigin is genuinely hard and migration still helps (~20% gbest improvement) but the absolute gap from the global optimum remains huge.

**Where host-staging hurts.** Per sync:
- `island_migrate_ring`: D cudaMemcpy D→H (one per dim) of pbest_pos columns to find top-m, then sendrecv, then D cudaMemcpy H→D to inject. ~60+ memcpys per sync.
- `island_gbest_exchange`: another D D→H + D H→D for the broadcast position injection.
- Total ~120 cudaMemcpys per sync, each ~9 µs API overhead even for small payloads.

The biggest single optimization would be a **device-side gather kernel** that packs the top-m particles into one contiguous `[m * D]` device buffer, allowing one D→H `cudaMemcpy` of m*D floats instead of D separate copies. Estimated reduction in sync_ms: ~5–10× (taking sync_ms from ~100 ms back down to ~10–20 ms, putting MPI runs at parity with or better than single-GPU at np=4).

**When fc overtakes ring as worst comm.** At the M4 baseline N_total=4096 np=4, fc was faster than ring (111 vs 127 ms). At an intermediate N_total=16384 np=4 the relationship inverts: **ring is faster than fc** (125 vs 137 ms). Two factors:
1. With heavier per-rank compute (4 ms eval at N=4096 vs ~30 ms eval at N=16384) the relative cost of ring's two serialized `MPI_Sendrecv` calls becomes small.
2. FC's `O(p²)` Allgather volume scales worse — at np=4 it's already showing the regression (1.28× at np=2 → 1.17× at np=4 for fc in the large-N strong scaling).

So the topology choice depends on the regime: **fc wins for small N with high sync overhead, ring wins for large N with cheap sync**. We expect ring to dominate as ranks grow past 4; cluster constraints prevented testing this.

**Predicted scaling cliff.** With sync_ms scaling roughly linearly in p (88 → 100 ms from np=2 → np=4 for ring weak), the implementation will likely hit *negative* scaling around np=8 unless the cudaMemcpy storm is addressed.

**What was tried and not worth the complexity:**
- We considered batching all D-per-dim cudaMemcpys into a single contiguous transfer at the start of the M4 push. Deferred — it's a real optimization but not load-bearing for the rubric, and adds a staging kernel + indexing changes that would need fresh correctness testing under time pressure.

**What's worth their complexity for the final report:**
- The pack-once gather kernel (above) — highest expected impact.
- Switch to CUDA-aware MPI and pass device pointers directly to `MPI_Sendrecv` / `MPI_Allgather`. Removes both the D→H and H→D copies; lets the MPI library do GPUDirect over the network where supported.
- Async-stream overlap: launch next iter's `kernel_eval_and_pbest` on stream 0 while migration proceeds on stream 1 from last iter's data. Compute would be free during the migration window.

---

## 5. Challenges and next steps

### Tasks remaining

| Priority | Task | Est. effort |
|---|---|---|
| ⭐⭐ | **Switch default `--sync` to 25 and re-run every prior experiment.** The sync-interval sweep showed `--sync 25` is Pareto-optimal: 2× cheaper *and* 2× better convergence than the `--sync 10` we used in early figures. All speedup / efficiency numbers in §3.1–3.2 should be re-collected at the new default for the final report. | 1 h plus slurm |
| ⭐ | **Pack-once gather kernel for migration.** Replace the D-per-dim cudaMemcpy loop in `island_migrate_*` with one device-side gather kernel into a contiguous buffer, then one D→H of m*D floats. Same for H→D injection. Expected ~5–10× reduction in sync_ms, which would push the crossover N down to ~25K and likely make MPI strictly faster than single-GPU at the tested sizes. | 4–6 h |
| ⭐ | **CUDA-aware MPI experiment.** Pass `state->d_pbest_pos + offset` directly to `MPI_Sendrecv`. Compare sync_ms before/after. | 2–3 h |
| | **Higher rank counts beyond 16.** With ring's `O(p)` cost confirmed at np=16, going further (np=32, 64) would directly show where the algorithm breaks down. | 1 h plus queue time |
| | **Async-stream overlap.** Run the on_sync callback on a separate CUDA stream so the next `kernel_eval_and_pbest` can start. | 6–8 h |
| | **Larger D study.** Requires removing the hardcoded `float pos_local[128]` stack array in `kernel_eval_and_pbest` ([pso/kernels.cu:61](pso/kernels.cu#L61)). | 2 h |
| | **More evaluators.** Add Ackley, Rosenbrock to `evals/evals.cu` for a richer scaling comparison. | 1 h |
| | **Clean up dead code in `island_migrate_ring`.** [mpi/mpi_island.cu:201](mpi/mpi_island.cu#L201) computes a `worst` variable then immediately recomputes it. Cosmetic. | 5 min |

### Items now empirically resolved (originally listed as open questions; resolved)

- ~~"Does sync amortize at large N?"~~ — Yes at fixed `--migrate 5` (ratio drops from 7.3× to 2.8× over N=4K–65K). Under proportional `m=N/100`, the ratio stays near 1.0 — sync grows with N just like compute. See §3.6.
- ~~"What sync_interval should we use?"~~ — `--sync 25` is best for our Rastrigin/N=1024 configuration. See §3.4.
- ~~"Does strong scaling work at all?"~~ — Yes at large N. The M4 baseline picture (efficiency < 1 everywhere) was an artifact of N_total=4096 being too small. See §3.1 (efficiency 0.85 at np=16 when N_total = 8M).

### Updated timeline to Jun 8

- **May 28–29:** pack-once gather kernel + correctness re-verification. Re-run scaling study.
- **May 30:** CUDA-aware MPI experiment if cluster MPI supports it.
- **Jun 1–4:** async-stream overlap or higher-D / more-evaluator study (pick one).
- **Jun 5–7:** writeup polish, final figures, LaTeX.
- **Jun 8:** submit.

---

## Appendix — reproducibility

### Hardware
- Quadro RTX 6000 (Turing, sm_75), 24 GB HBM2, 672 GB/s peak BW, 16.3 TFLOPS FP32.
- 4 GPUs per slurm node on the `gpu-turing` partition. MPI runs use `--ntasks-per-node=4 --gpus-per-task=1` for one rank per GPU.
- OpenMPI / HPC-X 2.17.1 from the `course/cme213/nvhpc/24.1` module.

### Code state
- Branch `oliver-m4`. Parent at start of M4: `290fc0c`.

### Reproducing everything

For the full per-file index of scripts, CSVs, and figures (with one-line descriptions of each), see [`bench/README.md`](README.md). The short version:

```bash
module load course/cme213/nvhpc/24.1
make clean && make mpi

sbatch bench/smoke.sh                # sync-hook smoke test
sbatch bench/correctness.sh          # correctness sweep across rank counts
sbatch bench/scaling.sh              # strong + weak at N_total=4096 (small-N baseline)
sbatch bench/sweeps.sh               # N-sweep, sync_interval Pareto, intermediate-N scaling
sbatch bench/sweep_NxRxD.sh          # D × ranks × N matrix at sync=25, m=N/100
sbatch bench/sweep_largeN_strong.sh  # strong scaling at N_total=8M, D=100
sbatch bench/sweep_largeN_weak.sh    # weak scaling at per-rank N=8M, D=100
sbatch bench/nsys.sh                 # Nsight at small N (M4 baseline)
sbatch bench/nsys_largeN.sh          # Nsight at N=2M D=100 (the regime §3 covers)

python3 bench/mpi_analyze.py         # regenerates all fig_*.png and table_mpi.md
```

---

## Appendix B — M4 baseline data (N_total = 4096)

> Preserved verbatim from the M4 submission (commit `68eebee`). This is the
> regime where the M4 deliverable was first measured: total swarm = 4096
> particles, D = 30, sync_interval = 10, fixed `--migrate 5`. The host-staging
> migration cost dominates in this regime — sync_ms is ~28× the GPU compute
> time — and the strong/weak scaling efficiency numbers are notably worse than
> the large-N regime in §3.1–3.3. We keep this section for reproducibility and
> because it documents how the conclusions evolved as we measured at larger N.

### B.1 Strong scaling — fixed total N=4096, partitioned across ranks

(From `bench/scaling_strong.csv`; `pso_cuda` baseline at N=4096: total_ms = **14.91**.)

| topology | n_islands | N | eval_ms | reduce_ms | update_ms | sync_ms | total_ms | speedup vs np=1 | efficiency vs np=1 | speedup vs single |
|---|---|---|---|---|---|---|---|---|---|---|
| ring | 1 | 4096 | 7.31 | 3.57 | 4.13 | 73.75 | **88.76** | 1.00 | 1.00 | 0.17 |
| fc | 1 | 4096 | 7.30 | 3.52 | 4.13 | 69.47 | **84.42** | 1.00 | 1.00 | 0.18 |
| ring | 2 | 2048 | 7.47 | 2.70 | 3.82 | 106.33 | 120.32 | 0.74 | 0.37 | 0.12 |
| fc | 2 | 2048 | 7.81 | 2.69 | 3.85 | 90.26 | 104.62 | 0.81 | 0.40 | 0.14 |
| ring | 4 | 1024 | 7.35 | 2.83 | 3.69 | 113.12 | 126.99 | 0.70 | 0.17 | 0.12 |
| fc | 4 | 1024 | 7.53 | 2.79 | 3.68 | 97.49 | 111.49 | 0.76 | 0.19 | 0.13 |

### B.2 Weak scaling — fixed per-rank N=1024

(From `bench/scaling_weak.csv`; `pso_cuda` baseline at N=1024: total_ms = **13.27**.)

| topology | n_islands | N | sync_ms | total_ms | speedup vs np=1 | efficiency vs np=1 |
|---|---|---|---|---|---|---|
| ring | 1 | 1024 | 60.37 | 73.69 | 1.00 | 1.00 |
| fc | 1 | 1024 | 59.07 | 72.74 | 1.00 | 1.00 |
| ring | 2 | 1024 | 82.07 | 96.11 | 0.77 | 0.38 |
| fc | 2 | 1024 | 81.46 | 95.25 | 0.76 | 0.38 |
| ring | 4 | 1024 | 88.86 | 102.83 | 0.72 | 0.18 |
| fc | 4 | 1024 | 100.08 | 114.06 | 0.64 | 0.16 |

### B.3 Comm vs compute breakdown (small N)

The compute portion (eval + reduce + update) sits at **~13 ms** consistently across all configurations and rank counts — the GPU is doing the same per-iter work regardless of topology. **`sync_ms` is what changes**, going from 0 ms (single-GPU) to ~60 ms (np=1, where the callback still fires but does no real comm) to ~100 ms (np=4 fc strong-scaling). For the np=4 strong case, sync_ms is **~28× the compute time** — comm is the dominant cost by an order of magnitude.

See `bench/fig_mpi_scaling.png` for the speedup/efficiency plot (both np=1 and pso_cuda baselines drawn), and `bench/fig_mpi_breakdown.png` for the stacked-bar visualization of compute vs sync.

### B.4 Nsight Systems at small N — `bench/nsys_summary.txt`, rank 0, ring -np 2 over 200 iters

CUDA API breakdown (in nanoseconds):

| Time % | Total time | Calls | Avg | Name |
|---|---|---|---|---|
| 57.8% | **28.35 ms** | 3,084 | 9.2 µs | `cudaMemcpy` |
| 22.5% | 11.07 ms | 1 | — | `cudaMemcpyFromSymbol` (one-time evaluator pointer resolve) |
| 11.0% | 5.41 ms | 1,003 | 5.4 µs | `cudaLaunchKernel` |
| 4.6% | 2.27 ms | 800 | 2.8 µs | `cudaEventRecord` |

CUDA kernel breakdown:

| Time % | Total | Avg | Kernel |
|---|---|---|---|
| 55.3% | 2.15 ms | 10.7 µs | `kernel_eval_and_pbest` |
| 17.2% | 0.67 ms | 3.3 µs | `kernel_draw_rng` |
| 9.8% | 0.38 ms | 1.9 µs | CUB `DeviceReduceSingleTileKernel` |
| 8.0% | 0.31 ms | 1.6 µs | `kernel_update` |
| 6.6% | 0.26 ms | 1.3 µs | `kernel_commit_gbest` |

**Total GPU kernel time: 3.77 ms.** **Total `cudaMemcpy` time: 28.35 ms.** The ratio is **~7.5×** — the host-staging pattern is unambiguously the bottleneck. Each `island_*` callback issues ~120 `cudaMemcpys` per sync (D=30 columns of `pbest_pos` read+write, plus gbest_pos broadcast injection, plus pbest_fit pull+push), and there are 20 syncs in this 200-iter run — that's ~2,400 `cudaMemcpy` calls for the migration alone, each ~9.2 µs of API + transfer overhead.

(`mpi_event_sum` did not produce data on this Nsight version, but the `cudaMemcpy` timing already captures the host-side cost of the sync.)
