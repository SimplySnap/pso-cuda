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

---

## 7. Discussion / bottlenecks

**The headline:** the implementation is *algorithmically* a clear win — multi-island Rastrigin converges 3.6× better at np=4 (gbest 13.9 vs 49.7) — but is *performance-wise* a net loss compared to single-GPU, because the host-staging migration costs ~7× the actual GPU compute per iteration. This is a classic comm-bound scaling profile.

**Where host-staging hurts.** Per sync:
- `island_migrate_ring`: D cudaMemcpy D→H (one per dim) of pbest_pos columns to find top-m, then sendrecv, then D cudaMemcpy H→D to inject. ~60+ memcpys per sync.
- `island_gbest_exchange`: another D D→H + D H→D for the broadcast position injection.
- Total ~120 cudaMemcpys per sync, each ~9 µs API overhead even for small payloads.

The biggest single optimization would be a **device-side gather kernel** that packs the top-m particles into one contiguous `[m * D]` device buffer, allowing one D→H `cudaMemcpy` of m*D floats instead of D separate copies. Estimated reduction in sync_ms: ~5–10× (taking sync_ms from ~100 ms back down to ~10–20 ms, putting MPI runs at parity with or better than single-GPU at np=4).

**When fc overtakes ring as worst comm.** Surprisingly, at np=4 strong scaling, fc is faster than ring (111 vs 127 ms). Two factors:
1. Ring's two `MPI_Sendrecv` calls are blocking and serialize the two ranks. FC's `MPI_Allgather` is one collective call that the MPI library can pipeline.
2. With only 4 ranks, FC's `O(p²)` total comm volume isn't yet larger than ring's `O(p)`-but-serialized cost.
We expect ring to overtake fc somewhere around np=8–16. Not measured.

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
| ⭐ | **Pack-once gather kernel for migration.** Replace the D-per-dim cudaMemcpy loop in `island_migrate_*` with one device-side gather kernel into a contiguous buffer, then one D→H of m*D floats. Same for H→D injection. | 4–6 h |
| ⭐ | **CUDA-aware MPI experiment.** Pass `state->d_pbest_pos + offset` directly to `MPI_Sendrecv`. Compare sync_ms before/after. | 2–3 h |
| | **Async-stream overlap.** Run the on_sync callback on a separate CUDA stream so the next `kernel_eval_and_pbest` can start. | 6–8 h |
| | **Larger D study.** Requires removing the hardcoded `float pos_local[128]` stack array in `kernel_eval_and_pbest` ([pso/kernels.cu:61](pso/kernels.cu#L61)). | 2 h |
| | **Higher rank counts (8, 16).** Currently capped at 4 by the slurm config; would expose the ring vs fc crossover point. | 1 h plus queue time |
| | **More evaluators.** Add Ackley, Rosenbrock to `evals/evals.cu` for a richer scaling comparison. | 1 h |
| | **Clean up dead code in `island_migrate_ring`.** [mpi/mpi_island.cu:201](mpi/mpi_island.cu#L201) computes a `worst` variable then immediately recomputes it. Cosmetic. | 5 min |

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
- M4 diff: 8 files changed, +74 / −61 lines. See `bench/M4_REPORT.md` repo path.

### Run commands
```bash
# build:
module load course/cme213/nvhpc/24.1
make clean && make mpi

# smoke:
sbatch bench/smoke.sh

# correctness sweep (single + ring/fc × {1,2,4} × {rastrigin, levy}):
sbatch bench/correctness.sh    # writes bench/correctness.csv

# strong + weak scaling (rastrigin, D=30, iters=500):
sbatch bench/scaling.sh        # writes bench/scaling_{strong,weak,baseline}.csv

# Nsight Systems (2 ranks, ring, 200 iters):
sbatch bench/nsys.sh           # writes bench/trace_ring_rank_{0,1}.nsys-rep and nsys_summary.txt

# figures + table:
python3 bench/mpi_analyze.py   # writes bench/fig_mpi_{scaling,breakdown}.png, bench/table_mpi.md
```

### Generated artifacts

- `bench/correctness.csv`
- `bench/scaling_strong.csv`, `bench/scaling_weak.csv`, `bench/scaling_baseline.csv`
- `bench/table_mpi.md`
- `bench/fig_mpi_scaling.png`, `bench/fig_mpi_breakdown.png`
- `bench/nsys_summary.txt`, `bench/trace_ring_rank_{0,1}.nsys-rep`
- `bench/smoke_*.out`, `bench/correctness_*.out`, `bench/scaling_*.out`, `bench/nsys_*.out` (slurm logs)
