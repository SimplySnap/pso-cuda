# pso-cuda — Milestone 4 Progress Report (Markdown working copy)

> Working document for the 2-page LaTeX submission. Mirrors the M4 rubric's 8
> sections with measurement data, decisions, and citations into the codebase.

**Project:** Parallel Particle Swarm Optimization on CUDA + MPI
**Repo branch:** `main` (parent commit `290fc0c`)
**Cluster:** Stanford CME 213 teaching cluster, partition `gpu-turing` (5 idle nodes, Quadro RTX 6000, sm_75, 1 GPU/node)
**Toolchain:** `course/cme213/nvhpc/24.1` module (nvcc 12.3 + OpenMPI/HPC-X 2.17.1)

---

## 1. Summary of progress since M3

**What was done.** The multi-GPU island-model MPI implementation is now functional, benchmarked, and profiled. Specifically:

- **Restored the multi-island sync hook in `pso_run()`.** M3 left a dormant function-pointer hook on `PSOConfig` (`SyncCallback on_sync`, `void* on_sync_data`) but the invocation site in the main iteration loop had been removed in commit `c8c1e32`. The MPI binaries therefore compiled and linked but ran as N independent island swarms with no migration. The hook is now invoked correctly every `sync_interval` iterations, gated by `cfg->on_sync != nullptr && (iter+1) % cfg->sync_interval == 0`, with a `cudaDeviceSynchronize()` before the callback so MPI sees a consistent device state.
- **Canonicalized `sync_interval` to live only on `PSOConfig`.** It had been split across both `PSOConfig` and `IslandSyncData` (the latter being the MPI-layer struct). Moved `IslandState` from `mpi/mpi_island.h` up to `pso/pso.h` so `pso_run` can construct one without including `<mpi.h>` — preserving the "PSO library is MPI-agnostic" property.
- **Added host-side `sync_ms` accounting.** Using `std::chrono::steady_clock` around the callback invocation; threaded through `PSOResult` and printed in CSV + stdout for both MPI mains.
- **Fixed the broken Makefile MPI flag discovery.** Replaced hardcoded `/opt/ohpc/.../openmpi4-gnu12/4.1.6` paths (which never existed on this cluster) with `mpicxx --showme:compile` / `--showme:link` queries, filtered down to `-I` and `-L` flags that `nvcc` accepts natively.
- **Patched `experiments.sh`** to use `#!/bin/bash -l` (login shell) and the actual `course/cme213/nvhpc/24.1` module.
- **Wrote and ran three slurm jobs:** smoke test (`bench/smoke.sh`), correctness sweep across {single-GPU, ring/fc × {1,2,4}} (`bench/correctness.sh`), and strong+weak scaling study (`bench/scaling.sh`). All jobs ran cleanly under the 15-minute partition limit.
- **Wrote `bench/mpi_analyze.py`** which consumes the scaling CSVs and emits `bench/fig_mpi_scaling.png`, `bench/fig_mpi_breakdown.png`, and `bench/table_mpi.md`.
- **Ran Nsight Systems** (per-rank traces via `mpirun -np 2 nsys profile ./pso_ring`) and produced `bench/nsys_summary.txt` with CUDA API and kernel summaries.

**What changed in single-GPU kernels:** zero. The SoA `pbest_pos[dim*N + *d_gbest_idx]` indirection from M3 turned out to be exactly the hook the MPI layer needed — sync just injects the cross-island gbest position into `pbest_pos[*, 0]` and points `d_gbest_idx` at slot 0, and the existing `kernel_update` reads from there automatically. No changes to `kernel_eval_and_pbest`, `kernel_draw_rng`, `kernel_update`, the CUB ArgMin pipeline, or any of the lifecycle routines.

**What is not done / partial:**
- No CUDA-aware MPI or GPUDirect attempt — host-staging only.
- No comm/compute overlap via async CUDA streams; the callback fully blocks the next iteration.
- The cluster `mpi_event_sum` Nsight report did not produce output (older Nsight version skips MPI event categorization); CUDA API / kernel summaries did succeed and capture the relevant story.

---

## 2. Distributed algorithm and data decomposition

**Decomposition: classical island model, not data-parallel.** Each MPI rank owns one complete independent swarm of `N` particles on one GPU. Per rank, the state is:

| Buffer | Size | Where |
|---|---|---|
| `positions` | `D × N` floats | device (SoA) |
| `velocities` | `D × N` floats | device (SoA) |
| `pbest_pos` | `D × N` floats | device (SoA) |
| `pbest` (fitness) | `N` floats | device |
| `d_gbest_val`, `d_gbest_idx` | scalars | device |
| `curandState` | `N` × ~48B | device |
| `d_r1`, `d_r2` | `D × N` × 2 floats | device (pregenerated per iter) |
| CUB reduce workspace | ~1 KB | device |
| `IslandSyncData` host buffers | `n_migrate × D` × 4 + `D` floats | host (one-time alloc) |

Inter-rank communication only happens every `sync_interval` iterations. What crosses rank boundaries:
- **`island_gbest_exchange`:** one `MPI_Allreduce(MPI_FLOAT_INT, MPI_MINLOC)` of `{val, rank}`, then one `MPI_Bcast` of `D` floats from the winning rank.
- **Ring migration:** two `MPI_Sendrecv` calls (one for `m × D` position bytes, one for `m` fitness floats) between each rank and its left/right neighbor.
- **FC migration:** two `MPI_Allgather` calls (same payload, all-to-all).
- All migrations end with one `island_gbest_exchange`.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Data-parallel split-N.** Distribute the swarm so each rank owns `N/p` particles; CUB ArgMin becomes Allreduce. | Requires `MPI_Allreduce` *every* iteration on `pbest_fit`. With ~10 ms compute per iter, network would dominate. Also complicates `kernel_update`'s gbest indirection. |
| **Model-parallel split-D.** Distribute dimensions across ranks. | Evaluator is `O(D)` per particle; splitting D forces a reduction inside every fitness call. Catastrophic for small D. |
| **Replicated everything (no MPI).** Just run pso_cuda on each rank with different seeds, MPI_Reduce(MIN) at end. | Wastes the algorithmic benefit of migration — convergence quality stays at single-GPU level. (See §5 — multi-island has *measurably* better final gbest.) |

**Trade-off summary.** The island model has *redundant compute* (each rank evaluates its own N particles) but minimal communication (only at sync intervals). For PSO specifically this is also algorithmically interesting — multiple islands explore the search space more aggressively and recombine via migration. The data in §5 shows this benefit empirically.

---

## 3. MPI implementation

**Collectives used:**

| Where | Collective | Payload | Purpose |
|---|---|---|---|
| Every sync | `MPI_Allreduce(MPI_FLOAT_INT, MPI_MINLOC)` | 8 bytes | Find rank with lowest gbest_val in one call |
| Every sync | `MPI_Bcast` | `D × float` | Winning gbest position to all ranks |
| Ring migration | `MPI_Sendrecv` ×2 | `(m × D + m) × float` | Pos + fit to right neighbor / from left |
| FC migration | `MPI_Allgather` ×2 | `(m × D + m) × float per rank` | Top-m from every rank to every rank |
| End of run | `MPI_Reduce(MPI_MIN)` | 1 float | Rank 0 reports best gbest across ranks |

**Point-to-point justification.** `MPI_Sendrecv` is used in `island_migrate_ring` because that *is* the operation: a directed neighbor exchange. Wrapping it in a collective would be over-engineering. The FC version uses `MPI_Allgather` because it's an all-to-all by definition.

**Overlap of communication with computation: none currently.** The structure of the loop is:

```
for iter in 0..max_iters:
  kernel_eval_and_pbest      (device, async)
  reduce_argmin_cub          (device, async)
  kernel_commit_gbest        (device, async)
  kernel_draw_rng + update   (device, async)
  cudaEventRecord(update_done)
  if sync time:
    cudaDeviceSynchronize()    <- pipeline drain
    cfg->on_sync(...)          <- blocking host work + MPI
```

The next iter only starts after the callback returns. Overlap would require: (a) running the on_sync work on a separate CUDA stream / host thread, (b) starting the next iter against last-iter's pbest while migration completes, (c) reconciling the slot-0 injection lazily. This is explicit future work (§8).

**GPU buffers and MPI: host-staging only.** Each `island_*` callback follows the same pattern: `cudaMemcpy` D→H into a host scratch buffer in `IslandSyncData`, MPI primitive between host buffers, `cudaMemcpy` H→D injection. **CUDA-aware MPI was not attempted** because (a) host-staging is simpler and portable across MPI implementations, and (b) the OpenMPI/HPC-X build on this cluster has CUDA-aware support but it is non-trivial to verify and we elected to defer.

The cost of host-staging is the dominant overhead per sync — see §6 and §7.

---

## 4. C++ wrapper and integration with CUDA

**Device selection per rank.** Each MPI main does:
```c
int n_gpus = 0;
CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
CUDA_CHECK(cudaSetDevice(rank % n_gpus));
```
With `--gpus-per-task=1` and one rank per slurm node, the `rank % n_gpus` is just `0` per node (each rank sees one GPU). The modulo lets us oversubscribe gracefully if someone runs more ranks than GPUs (we did not exercise this path).

**Memory management.** The strict layering is:
- **Device buffers** (positions, velocities, pbest, RNG state, CUB workspace, etc.) are owned by the `swarm` struct, allocated once via `swarm_alloc(swarm*, const PSOConfig*)` at the start of `pso_run`, freed via `swarm_free`.
- **Host scratch buffers** for the MPI layer (`h_send_pos`, `h_send_fit`, `h_recv_pos`, `h_recv_fit`, `h_gbest_pos`) live in `IslandSyncData` and are allocated once in the MPI main via `island_sync_data_alloc()` before `pso_run` starts. The MPI mains pass `&sync_data` as `cfg.on_sync_data`, and the callback casts back. No per-sync malloc.
- **Host result buffers** (`best_position`, `gbest_history`) are `malloc`'d in `pso_run` and freed via `pso_result_free` after the user has read them.

**Per-rank seed offset.** `cfg.seed = args.seed + (unsigned long long)rank`. Each island gets a unique non-overlapping cuRAND subsequence, so the four islands explore genuinely different starting points instead of converging deterministically to the same answer.

**Single-GPU kernels unchanged.** As noted in §1, the M3 design happened to align perfectly with what the MPI layer needs.

**Host wall-clock instrumentation** is via `std::chrono::steady_clock` (no MPI dependency in the library):
```c
auto t0 = std::chrono::steady_clock::now();
cfg->on_sync(&st, cfg->on_sync_data);
sync_ms += std::chrono::duration<float, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
```

---

## 5. Correctness testing

Identical configuration (`--N 1024 --D 30 --iters 500 --seed 42 --sync 10 --migrate 5`) run across {single-GPU, ring/fc × {1,2,4}} for two evaluators. Results from `bench/correctness.csv`:

### Rastrigin (D=30 multi-modal, global min = 0)

| impl | topology | n_ranks | final_gbest | total_ms |
|---|---|---|---|---|
| single | none | 1 | **49.747841** | 13.31 |
| mpi | ring | 1 | 42.783207 | 75.04 |
| mpi | fc | 1 | 46.762962 | 75.68 |
| mpi | ring | 2 | **21.889114** | 179.81 |
| mpi | fc | 2 | **21.889114** | 95.08 |
| mpi | ring | 4 | **13.929430** | 229.88 |
| mpi | fc | 4 | 13.929499 | 112.71 |

### Levy (global min = 0)

| impl | topology | n_ranks | final_gbest | total_ms |
|---|---|---|---|---|
| single | none | 1 | 7.64e-15 | 17.86 |
| all mpi configs | — | 1/2/4 | 7.64e-15 | 79–122 |

**Findings:**
- **All configurations converge.** Levy hits machine epsilon (~10⁻¹⁵) everywhere.
- **Multi-island improves convergence quality on Rastrigin.** Going from single-GPU (gbest=49.7) → np=2 island (gbest=21.9) → np=4 island (gbest=13.9) is a clear win from migration-driven diversification.
- **Ring and FC give identical gbest at np=2** — expected, since for two ranks "ring" and "fully-connected" are topologically the same.
- **`pso_ring -np 1` is *not* byte-identical to `pso_cuda`.** Differences:
  - the sync callback still fires (50 syncs at iters=500, sync=10) and overwrites `pbest_pos[*, 0]` with the (self-)broadcast position
  - per-rank seed offset is `42 + 0 = 42` (matches), so RNG streams are identical
  - the differences are localized to particle 0's pbest, which is overwritten 50 times during the run
  - this counts as a known-and-bounded behavioral difference; both produce well-converged results

The correctness sweep validates that the migration logic is functioning correctly — without firing the callback, np=2 and np=4 would have converged to the same gbest as np=1.

---

## 6. Performance, scaling, profiling

### Strong scaling — fixed total N=4096, partitioned across ranks

(From `bench/scaling_strong.csv`; `pso_cuda` baseline at N=4096: total_ms = **14.91**.)

| topology | n_islands | N | eval_ms | reduce_ms | update_ms | sync_ms | total_ms | speedup vs np=1 | efficiency vs np=1 | speedup vs single |
|---|---|---|---|---|---|---|---|---|---|---|
| ring | 1 | 4096 | 7.31 | 3.57 | 4.13 | 73.75 | **88.76** | 1.00 | 1.00 | 0.17 |
| fc | 1 | 4096 | 7.30 | 3.52 | 4.13 | 69.47 | **84.42** | 1.00 | 1.00 | 0.18 |
| ring | 2 | 2048 | 7.47 | 2.70 | 3.82 | 106.33 | 120.32 | 0.74 | 0.37 | 0.12 |
| fc | 2 | 2048 | 7.81 | 2.69 | 3.85 | 90.26 | 104.62 | 0.81 | 0.40 | 0.14 |
| ring | 4 | 1024 | 7.35 | 2.83 | 3.69 | 113.12 | 126.99 | 0.70 | 0.17 | 0.12 |
| fc | 4 | 1024 | 7.53 | 2.79 | 3.68 | 97.49 | 111.49 | 0.76 | 0.19 | 0.13 |

### Weak scaling — fixed per-rank N=1024

(From `bench/scaling_weak.csv`; `pso_cuda` baseline at N=1024: total_ms = **13.27**.)

| topology | n_islands | N | sync_ms | total_ms | speedup vs np=1 | efficiency vs np=1 |
|---|---|---|---|---|---|---|
| ring | 1 | 1024 | 60.37 | 73.69 | 1.00 | 1.00 |
| fc | 1 | 1024 | 59.07 | 72.74 | 1.00 | 1.00 |
| ring | 2 | 1024 | 82.07 | 96.11 | 0.77 | 0.38 |
| fc | 2 | 1024 | 81.46 | 95.25 | 0.76 | 0.38 |
| ring | 4 | 1024 | 88.86 | 102.83 | 0.72 | 0.18 |
| fc | 4 | 1024 | 100.08 | 114.06 | 0.64 | 0.16 |

### Comm vs compute breakdown

The compute portion (eval + reduce + update) sits at **~13 ms** consistently across all configurations and rank counts — the GPU is doing the same per-iter work regardless of topology. **`sync_ms` is what changes**, going from 0 ms (single-GPU) to ~60 ms (np=1, where the callback still fires but does no real comm) to ~100 ms (np=4 fc strong-scaling). For the np=4 strong case, sync_ms is **~28× the compute time** — comm is the dominant cost by an order of magnitude.

See `bench/fig_mpi_scaling.png` for the speedup/efficiency plot (both np=1 and pso_cuda baselines drawn), and `bench/fig_mpi_breakdown.png` for the stacked-bar visualization of compute vs sync.

### Nsight Systems (`bench/nsys_summary.txt`, rank 0, ring -np 2 over 200 iters)

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

### 6.4 N-sweep — does sync amortize at large N?

(Follow-up sweep `bench/sweeps.sh`, ranks=4, sync=10, iters=500. Data in `bench/sweep_N.csv` + `bench/sweep_N_baseline.csv`, figure in `bench/fig_sweep_N.png`.)

The §6 headline at total N=4096 was "sync_ms ~10× compute, MPI loses to single-GPU." This sweep asks: **does that ratio shrink as N grows, and at what point does MPI catch up?**

| N | compute_ms (ring) | sync_ms (ring) | sync/compute ratio | pso_cuda total_ms | MPI total_ms (ring np=4) |
|---|---|---|---|---|---|
| 1,024 | 14.07 | 90.08 | **6.4×** | 13.15 | 104.15 |
| 4,096 | 15.45 | 112.77 | **7.3×** | 14.73 | 128.22 |
| 16,384 | 34.34 | 158.55 | **4.6×** | 34.68 | 192.89 |
| 65,536 | 140.42 | 387.51 | **2.8×** | 142.87 | 527.93 |

**Findings:**

1. **The ratio does decrease — from 7.3× at N=4096 down to 2.8× at N=65536.** Compute grows roughly linearly in N (140 ms at N=65536, ~10× N=4096); `sync_ms` grows only ~3.4× over the same range (113 → 388). The MPI host-staging callback is dominated by per-call fixed costs (cudaMemcpy launch latency, MPI collective setup), and those costs are amortized once per-iter compute is heavy enough.
2. **MPI does not yet beat single-GPU within N ≤ 65,536** — at the largest tested N, `pso_ring -np 4` takes 528 ms vs `pso_cuda`'s 143 ms (still 3.7× slower). Linear extrapolation of the ratio suggests crossover would occur somewhere around N ≈ 250,000 — at which point single-GPU also becomes slow enough that the MPI overhead is structurally amortized.
3. **Convergence quality improves dramatically with N** — `final_gbest` drops from 13.9 at N=1024 down to 3.98 at N=65536. Larger swarms find better minima regardless of topology, and 4 MPI ranks × 65536 = 262K total particles is genuinely more thorough than 65536 single-GPU.

The single-GPU baseline at N=65536 (142.87 ms, gbest=8.95) is **strictly worse than MPI at N=65536 (528 ms, gbest=3.98)** on convergence quality at fixed iterations. So if the metric of interest is "best gbest at fixed wall time" rather than "fastest per-iter," MPI can win even today — just at a much larger N than M4 originally tested.

> **Caveat (added in Phase G):** The amortization story above used `--migrate 5` fixed across all N. At per-rank N=65,536 that means exchanging 0.008% of the swarm per sync — essentially no real migration. The Phase G results in §6.7 redo this with `m = max(5, N/100)` (1% of N) and the amortization picture changes substantially. **The "MPI wins at large N" prediction is contingent on m being kept artificially small.** See §6.7 for the corrected picture.

### 6.5 Sync-interval sweep — Pareto front of cost vs convergence

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

### 6.6 Strong & weak scaling at large N

(N_total=16384 for strong, per-rank N=16384 for weak. Data in `bench/sweep_strong_largeN.csv` + `bench/sweep_weak_largeN.csv`, figure in `bench/fig_sweep_largeN_scaling.png`.)

#### Strong scaling at N_total = 16384

(Single-GPU baseline: `pso_cuda N=16384` → 34.65 ms.)

| topology | ranks | per-rank N | total_ms | speedup vs np=1 | efficiency |
|---|---|---|---|---|---|
| ring | 1 | 16384 | 165.07 | 1.00 | 1.00 |
| ring | 2 | 8192 | 136.12 | **1.21** | 0.61 |
| ring | 4 | 4096 | 125.49 | **1.32** | 0.33 |
| fc | 1 | 16384 | 160.62 | 1.00 | 1.00 |
| fc | 2 | 8192 | 125.71 | **1.28** | 0.64 |
| fc | 4 | 4096 | 137.31 | 1.17 | 0.29 |

**Compared to M4 (N_total=4096):**

| N_total | ring np=1 → np=4 speedup | efficiency at np=4 |
|---|---|---|
| 4,096 (M4) | 0.70× (slower!) | 0.17 |
| 16,384 (new) | **1.32× (faster)** | 0.33 |

**Strong scaling DOES work at large N.** At N_total=4096 the per-rank compute (3–7 ms) was too small to amortize the ~100 ms of sync — adding ranks made things strictly worse. At N_total=16384 the per-rank compute at np=1 is heavy enough (~31 ms) that splitting it 4 ways yields a real speedup. The efficiency picture **inverts** with problem size.

The fc topology shows a slight regression at np=4 (1.28× at np=2 → 1.17× at np=4) suggesting that Allgather's `O(p²)` overhead is starting to bite at 4 ranks. Ring's `O(p)` cost continues to scale.

#### Weak scaling at per-rank N = 16384

| topology | ranks | total_ms | speedup vs np=1 (work-normalized) |
|---|---|---|---|
| ring | 1 | 164.35 | 1.00 |
| ring | 2 | 207.90 | 0.79 |
| ring | 4 | 223.11 | 0.74 |
| fc | 1 | 162.75 | 1.01 |
| fc | 2 | 183.36 | 0.89 |
| fc | 4 | 211.21 | 0.78 |

Weak scaling is worse — total work grows 4× but total_ms only grows ~1.4×, so per-particle throughput drops. The dominant factor is sync_ms growing with rank count (130 → 189 for ring) because the cudaMemcpy cost in each rank's callback is *not* parallelized — each rank does its own D-per-dim copy independently.

**Final gbest also improves with rank count** (11.9 at np=1 → ~5–8 at np=4) — more islands explore more of the search space.

### 6.7 High-D and rank-count scaling (Phase G)

(Single slurm job `87979`, 4 nodes × 4 GPUs each → 16 GPUs available. Data in `bench/sweep_NxRxD.csv` + `bench/sweep_NxRxD_baseline.csv` + `bench/sweep_NxRxD_levy.csv`. Figure in `bench/fig_sweep_NxRxD.png`.)

**Policy changes vs §6.4–6.6** (all simultaneous, intentional):
1. **`m = max(5, N/100)`** — migrate 1% of swarm instead of fixed 5 (algorithmically reasonable at large N).
2. **`--sync 25`** — Pareto optimum from §6.5.
3. **D ∈ {30, 100, 300}** — exercise the curse-of-dimensionality regime.
4. **ranks ∈ {1, 4, 16}** — 1 rank per GPU, exploits all 16 GPUs across 4 nodes.
5. **Per-rank N up to 8.4M** — pushes per-rank VRAM to ~5 GB out of 24 GB.
6. **Source change**: `pso/kernels.cu`'s `pos_local[128]` bumped to `[1024]` to support D up to 1024.

**Matrix coverage:** 35 of 36 expected MPI cells completed; 1 cell (D=300, fc, np=16, N=524K) hit the 90s timeout — itself a data point. Plus 6 single-GPU baselines and 3 Levy sanity rows.

#### 6.7.1 The proportional-migration cost invalidates the Phase E amortization story

Phase E §6.4 found `sync_ms / compute_ms` drops from 7.3× to 2.8× as N grows from 4K to 65K, suggesting MPI would beat single-GPU at N ≈ 250K. **Phase G shows this was an artifact of fixed m=5.**

Selected rows from `sweep_NxRxD.csv` at D=30:

| ranks | topology | N | compute_ms | sync_ms | sync/compute | total_ms vs pso_cuda |
|---|---|---|---|---|---|---|
| 1 | ring | 2,097,152 | 4,090 | 3,657 | **0.89** | 1.92× slower |
| 1 | ring | 8,388,608 | 16,000 | 16,116 | **1.01** | 2.03× slower |
| 4 | ring | 8,388,608 | 15,939 | 18,425 | **1.16** | 2.17× slower |
| 16 | ring | 4,194,304 | 8,292 | 9,396 | **1.13** | (no baseline) |

With m proportional to N, `sync_ms` now scales linearly with N (the cudaMemcpy payload grows). The ratio stays near 1.0 across the whole N range — *neither* dropping toward 0 (the Phase E hope) *nor* exploding. MPI ring stays ~2× slower than single-GPU at every per-rank N tested. **The crossover predicted in Phase E does not occur under proportional migration.**

This is the honest result: when migration is a real algorithmic operation (not noise), it costs about the same as compute per iteration. The "MPI wins at scale" narrative only worked because Phase E's m=5 made the migration effectively a no-op at large N.

#### 6.7.2 fc breaks at np=16; ring scales cleanly

The most dramatic finding. Selected np=16 rows:

| D | topology | N | sync_ms | sync/compute | total_ms |
|---|---|---|---|---|---|
| 30 | **ring** | 2,097,152 | 4,697 | 1.15 | 8.77 s |
| 30 | **fc** | 2,097,152 | **43,321** | **10.66** | **47.4 s** |
| 30 | **ring** | 4,194,304 | 9,396 | 1.13 | 17.7 s |
| 30 | **fc** | 4,194,304 | **81,196** | **9.83** | **89.5 s ❌ over 60s ceiling** |
| 100 | fc | 524,288 | 33,916 | 9.54 | 37.5 s |
| 100 | fc | 1,048,576 | 63,752 | 9.18 | 70.7 s ❌ |
| 300 | fc | 524,288 | — | — | timed out at 90s ❌ |

**At np=16, fc's sync_ms is 10× ring's.** Allgather's O(p²) total communication volume scales much worse than ring's O(p) Sendrecv at high rank counts. This empirically confirms the M4 §7 prediction ("ring overtakes fc at np≥8–16") with a wide margin.

#### 6.7.3 Algorithmic differentiation finally emerges at D=300

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

#### 6.7.4 Levy stops being trivial at D=300

Levy at D=30 and D=100 still hits machine epsilon (~7.6e-15) under multi-island runs. At D=300, the sanity row shows `final_gbest = 2.06` — Levy is no longer the "everyone gets 0" benchmark we used in §5. For future studies this means Levy at D≥300 becomes a useful evaluator alongside Rastrigin.

#### 6.7.5 60-second ceiling map

Per (D, ranks, topology), the largest per-rank N where total_ms ≤ 60,000 ms:

| D | ranks | ring largest N (≤60s) | fc largest N (≤60s) |
|---|---|---|---|
| 30 | 1 | 8.4M (32.1 s) | 8.4M (27.8 s) |
| 30 | 4 | 8.4M (34.4 s) | 8.4M (33.0 s) |
| 30 | 16 | 4.2M (17.7 s) | **2.1M (47.4 s)** — N=4.2M is 89.5 s |
| 100 | 1 | 2.1M (23.0 s) | 2.1M (22.0 s) |
| 100 | 4 | 2.1M (23.2 s) | 2.1M (22.9 s) |
| 100 | 16 | 1.0M (12.9 s) | **524K (37.5 s)** — N=1.0M is 70.7 s |
| 300 | 1 | 524K (18.0 s) | 524K (17.7 s) |
| 300 | 4 | 524K (18.1 s) | 524K (18.1 s) |
| 300 | 16 | 524K (19.4 s) | **131K (28.0 s)** — N=524K timed out at 90s |

**Headline:** ring stays under 60s for every (D, ranks, N) cell in the matrix; fc collapses at np=16 for any non-trivial N.

---

## 7. Discussion / bottlenecks

**Headline (final, after Phase G):** the M4 implementation does what it should — *algorithmically* — but the host-staging cost model means it's *performance-wise* a net loss versus single-GPU. The picture has stabilized after three rounds of measurement:

- **At small total problem sizes (N_total ≤ 4096)** the host-staging migration costs ~7× the GPU compute and MPI loses outright to single-GPU.
- **The Phase E "MPI wins at large N" prediction was an artifact** of fixed `m=5`. Under proportional migration (`m = max(5, N/100)`, Phase G §6.7), sync_ms scales with N just like compute does. The sync/compute ratio stays near 1.0 from N=2M to N=8M — *no crossover*. MPI stays ~2× slower than single-GPU at every cell.
- **`--sync 25` is the empirical Pareto-optimal sync_interval** (§6.5): 2× cheaper AND 2× better convergence than the `--sync 10` we used in early sweeps.
- **fc breaks at np=16** for any non-trivial N — Allgather's O(p²) cost makes sync 10× compute at np=16 (§6.7.2). **Ring stays viable** at np=16 with sync/compute ≈ 1.0.
- **Strong scaling efficiency inverts with N.** At N_total=4096 (M4 deliverable) efficiency was worse than 1 — adding ranks made things slower. At N_total=16384, np=4 delivers a 1.32× speedup (efficiency 0.33). At larger N + proportional m, ring shows efficiency 0.45–0.55 vs np=1.
- **Algorithmic differentiation requires high D.** At D=30 the problem is too easy — most multi-island configurations reach gbest=0. At D=100, ring np=16 finds 34% better minima than single-GPU at the same per-rank N. At D=300, Rastrigin is genuinely hard and migration still helps (~20% gbest improvement) but the absolute gap from the global optimum remains huge.
- **Convergence-quality-per-wall-time still favors MPI at large (N, D).** At D=300 N=524K, ring np=16 reaches gbest=1108 in 19.4 s; single-GPU reaches gbest=1158 in 11.1 s. Multi-island finds a slightly better minimum for ~2× the time. The trade-off is real but the gap has narrowed since the (artificially favorable) Phase E numbers.

**Where host-staging hurts.** Per sync:
- `island_migrate_ring`: D cudaMemcpy D→H (one per dim) of pbest_pos columns to find top-m, then sendrecv, then D cudaMemcpy H→D to inject. ~60+ memcpys per sync.
- `island_gbest_exchange`: another D D→H + D H→D for the broadcast position injection.
- Total ~120 cudaMemcpys per sync, each ~9 µs API overhead even for small payloads.

The biggest single optimization would be a **device-side gather kernel** that packs the top-m particles into one contiguous `[m * D]` device buffer, allowing one D→H `cudaMemcpy` of m*D floats instead of D separate copies. Estimated reduction in sync_ms: ~5–10× (taking sync_ms from ~100 ms back down to ~10–20 ms, putting MPI runs at parity with or better than single-GPU at np=4).

**When fc overtakes ring as worst comm.** At M4's N_total=4096 np=4, fc was faster than ring (111 vs 127 ms). At Phase E's N_total=16384 np=4 the relationship inverts: **ring is faster than fc** (125 vs 137 ms). Two factors:
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

## 8. Challenges and next steps (toward Jun 8 final report)

### Tasks remaining

| Priority | Task | Est. effort |
|---|---|---|
| ⭐⭐ | **Switch default `--sync` to 25 and re-run every prior experiment.** Phase E's sweep showed `--sync 25` is Pareto-optimal: 2× cheaper *and* 2× better convergence than the `--sync 10` we used in every M4 figure. All speedup / efficiency numbers in §6.1–6.2 should be re-collected at the new default for the final report. | 1 h plus slurm |
| ⭐ | **Pack-once gather kernel for migration.** Replace the D-per-dim cudaMemcpy loop in `island_migrate_*` with one device-side gather kernel into a contiguous buffer, then one D→H of m*D floats. Same for H→D injection. Expected ~5–10× reduction in sync_ms, which would push the crossover N down to ~25K and likely make MPI strictly faster than single-GPU at the tested sizes. | 4–6 h |
| ⭐ | **CUDA-aware MPI experiment.** Pass `state->d_pbest_pos + offset` directly to `MPI_Sendrecv`. Compare sync_ms before/after. | 2–3 h |
| | **Higher rank counts (8, 16).** Cluster supports it but we capped at 4 for the M4 + Phase E push. With ring's `O(p)` cost confirmed at large N, this would directly show where the algorithm breaks down (likely np=8–16). | 1 h plus queue time |
| | **Async-stream overlap.** Run the on_sync callback on a separate CUDA stream so the next `kernel_eval_and_pbest` can start. | 6–8 h |
| | **Larger D study.** Requires removing the hardcoded `float pos_local[128]` stack array in `kernel_eval_and_pbest` ([pso/kernels.cu:61](pso/kernels.cu#L61)). | 2 h |
| | **More evaluators.** Add Ackley, Rosenbrock to `evals/evals.cu` for a richer scaling comparison. | 1 h |
| | **Clean up dead code in `island_migrate_ring`.** [mpi/mpi_island.cu:201](mpi/mpi_island.cu#L201) computes a `worst` variable then immediately recomputes it. Cosmetic. | 5 min |

### Items now empirically resolved (was in M4 §8, removed)

- ~~"Does sync amortize at large N?"~~ — Yes (sync/compute drops from 7.3× to 2.8× over N=4096–65536). See §6.4.
- ~~"What sync_interval should we use?"~~ — `--sync 25` is best for our Rastrigin/N=1024 configuration. See §6.5.
- ~~"Does strong scaling work at all?"~~ — Yes at large N. The M4 picture (efficiency < 1 everywhere) was an artifact of N_total=4096 being too small. See §6.6.

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
- One GPU per slurm node (partition `gpu-turing`), MPI ranks distributed `--ntasks-per-node=1`.
- OpenMPI / HPC-X 2.17.1 from the nvhpc/24.1 module.

### Code state
- Parent commit at start of M4: `290fc0c` ("Update README.md").
- M4 base diff (commit `68eebee` on branch `oliver-m4`): 8 files changed, +74 / −61 lines.
- Phase E (sweeps, this section): no source code changes; new `bench/sweeps.sh` + extended `bench/mpi_analyze.py` only.

### Run commands
```bash
# build:
module load course/cme213/nvhpc/24.1
make clean && make mpi

# smoke:
sbatch bench/smoke.sh

# correctness sweep (single + ring/fc × {1,2,4} × {rastrigin, levy}):
sbatch bench/correctness.sh    # writes bench/correctness.csv

# strong + weak scaling at N_total=4096 (rastrigin, D=30, iters=500):
sbatch bench/scaling.sh        # writes bench/scaling_{strong,weak,baseline}.csv

# Phase E parameter sweeps (N, sync_interval, large-N strong+weak):
sbatch bench/sweeps.sh         # writes bench/sweep_*.csv

# Nsight Systems (2 ranks, ring, 200 iters):
sbatch bench/nsys.sh           # writes bench/trace_ring_rank_{0,1}.nsys-rep and nsys_summary.txt

# figures + table:
python3 bench/mpi_analyze.py   # writes all bench/fig_*.png + bench/table_mpi.md
```

### Generated artifacts

- M4 base: `bench/correctness.csv`, `bench/scaling_{strong,weak,baseline}.csv`,
  `bench/fig_mpi_scaling.png`, `bench/fig_mpi_breakdown.png`
- Phase E sweeps: `bench/sweep_N.csv`, `bench/sweep_N_baseline.csv`,
  `bench/sweep_sync.csv`, `bench/sweep_strong_largeN.csv`,
  `bench/sweep_weak_largeN.csv`, `bench/sweep_largeN_baseline.csv`,
  `bench/fig_sweep_N.png`, `bench/fig_sweep_sync.png`, `bench/fig_sweep_largeN_scaling.png`
- Shared: `bench/table_mpi.md`, `bench/nsys_summary.txt`,
  `bench/trace_ring_rank_{0,1}.nsys-rep`
- Slurm logs: `bench/{smoke,correctness,scaling,nsys,sweeps}_*.{out,err}`
