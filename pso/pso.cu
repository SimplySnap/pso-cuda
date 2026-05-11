
/*
Kernel structure:
┌─────────────────────────────────────────────────────┐
│  For each iteration:                                │
│                                                     │
│  1. EVALUATE       fitness[i] = f(positions[i])     │
│     1 thread per particle                           │
│                                                     │
│  2. UPDATE PBEST   if fitness[i] < pbest_fit[i]:    │
│                        pbest_pos[i] = positions[i]  │
│                        pbest_fit[i] = fitness[i]    │
│     Can fuse with step 1                            │
│                                                     │
│  3. REDUCE GBEST   gbest = min over all particles   │
│     Parallel reduction → single warp shuffle or    │
│     thrust::min_element                             │
│                                                     │
│  4. UPDATE V & X   per-element velocity + position  │
│     1 thread per (particle × dim)                  │
│     Needs RNG — cuRAND state per thread             │
└─────────────────────────────────────────────────────┘
*/

//meow


//block finds block-level best with cheap SMEM redux
//if cur better than gbest -> add fitness & positionto atomicAdd to shared queue
//then, read aux[] array, par tree redux over blocks to find true global best
//thread 0 block secretary
//

/*
pso.cu
│
├── swarm_alloc()       — cudaMalloc all arrays + reduce workspace
├── swarm_free()        — cudaFree everything
├── swarm_init()        — cuRAND init, random positions/velocities
│
├── pso_run()           ← main loop
│     │
│     ├── kernel_eval_and_pbest<<<>>>()    [kernels.cuh]
│     │
│     ├── reduce_argmin_cub()              [reduce.cuh / reduce.cu]
│     │     └── swap to reduce_argmin_custom() here with no other changes
│     │
│     ├── copy gbest_pos from positions[gbest_idx]
│     │
│     └── kernel_update<<<>>>()            [kernels.cuh]
│
└── pso_result_free()
*/

// =============================================================================
// Milestone 3 — single-GPU implementation TODOs
// =============================================================================
//
// --- KERNELS (move to pso/kernels.cuh + pso/kernels.cu) ----------------------
//
// TODO(M3): __global__ kernel_curand_init(curandState* states, ull seed, int n)
//           One thread per RNG slot. Run once during swarm_init.
//
// TODO(M3): __global__ kernel_eval_and_pbest(
//               const float* positions,   // [D*N] SoA
//               float*       fitness,     // [N]
//               float*       pbest,       // [N]
//               float*       pbest_pos,   // [D*N]
//               EvaluatorFn  f,
//               int N, int D)
//           - 1 thread per particle, grid-stride loop over N.
//           - Each thread gathers its D coords (strided reads from SoA), calls f.
//           - If fit < pbest[i]: write pbest[i] = fit, copy positions slice -> pbest_pos.
//           - Fuses eval + pbest update (one pass over particle state).
//
// TODO(M3): __global__ kernel_update(
//               float* positions, float* velocities,
//               const float* pbest_pos, const float* gbest_pos,
//               curandState* states,
//               float w, float c1, float c2,
//               float bound_lo, float bound_hi,
//               int N, int D)
//           - Warp-per-dimension, thread-per-particle (Idea 2 from README).
//           - threadIdx.x indexes particle; warp/blockIdx.y indexes dim slice.
//           - Draw r1, r2 from per-thread curandState.
//           - v = w*v + c1*r1*(pbest_pos - pos) + c2*r2*(gbest_pos - pos)
//           - pos = clamp(pos + v, bound_lo, bound_hi)
//           - gbest_pos[d] is broadcast — load via __ldg or stage in shared mem.
//
// --- REDUCTION (move to pso/reduce.cuh + pso/reduce.cu) ----------------------
//
// TODO(M3): struct ReduceResult { float val; int idx; };  // referenced in pso.h
//
// TODO(M3): reduce_argmin_cub(const float* pbest, int N,
//                             void* tmp, size_t tmp_bytes,
//                             ReduceResult* d_out, cudaStream_t s)
//           - Wraps cub::DeviceReduce::ArgMin.
//           - Use the two-call idiom in swarm_alloc to size tmp_bytes.
//
// TODO(M3): reduce_argmin_custom(...)
//           - L1: warp shuffle butterfly carrying (fit, idx) pair via
//                 __shfl_down_sync(0xFFFFFFFF, ...).
//           - L2: warp leaders write to shared mem; one warp reduces across warps.
//           - L3 (optional, large grids): staging array of block winners +
//                 second kernel for final argmin (avoid atomicCAS contention).
//
// TODO(M3): __global__ kernel_copy_gbest_pos(
//               const float* pbest_pos, float* gbest_pos,
//               const ReduceResult* d_in, int N, int D)
//           - D threads gather gbest_pos[d] = pbest_pos[d*N + d_in->idx].
//           - Only update if d_in->val < current gbest_val (host-side check OK).
//
// --- LIFECYCLE (this file) ---------------------------------------------------
//
// TODO(M3): swarm_alloc()
//           - cudaMalloc: positions, velocities, pbest_pos, pbest, fitness,
//             gbest_pos[D], d_reduce_out, curandState pool.
//           - Query CUB workspace size, cudaMalloc reduce_tmp.
//           - All allocations checked via CUDA_CHECK macro.
//
// TODO(M3): swarm_free()
//           - cudaFree every pointer above, set to nullptr.
//
// TODO(M3): swarm_init(seed)
//           - Launch kernel_curand_init.
//           - Launch init kernel to fill positions/velocities and seed pbest=+INF.
//           - Seed host gbest_val = +INF, gbest_idx = -1.
//           - IMPORTANT: pbest_pos does NOT need explicit init IF the first
//             eval_and_pbest pass uses `<` against pbest=+INF (always true) and
//             unconditionally copies positions->pbest_pos on that first hit.
//             If the comparison is `<=` or skips on equality, you'll read
//             uninitialized pbest_pos on iter 1. Pick a convention and stick
//             to it in kernel_eval_and_pbest.
//
// TODO(M3): pso_run() main loop
//           for it = 0..max_iters-1:
//             1. kernel_eval_and_pbest<<<grid, block>>>
//             2. reduce_argmin_*  -> d_reduce_out
//             3. (host or tiny kernel) if d_reduce_out.val < gbest_val:
//                    update gbest_val/gbest_idx, launch kernel_copy_gbest_pos
//             4. kernel_update<<<grid2, block2>>>
//           cudaMemcpy gbest_pos D->H into PSOResult.best_position.
//
// TODO(M3): gbest update path — pick one and commit:
//   (a) cudaMemcpy d_reduce_out -> host (8 bytes) every iter, branch on host,
//       then launch kernel_copy_gbest_pos if improved.
//       + Simple, easy to log per-iter gbest for the convergence plot.
//       - Forces an implicit device sync every iter (bad for kernel overlap).
//   (b) Single-thread device kernel "kernel_commit_gbest" that reads
//       d_reduce_out, compares to a device-resident gbest_val, conditionally
//       updates it, and runs the copy_gbest_pos gather inline.
//       + Zero host sync in the inner loop; better for benchmarking.
//       - Need a separate path to dump gbest_history to host at end.
//   Recommend (b) + record gbest_history[it] from inside that same kernel.
//
// TODO(M3): gbest history logging — append d_gbest_history[it] = current
//   gbest_val from inside the commit kernel (option b above). Copy the full
//   array D->H once at end of pso_run for the convergence-vs-iter figure.
//
// TODO(M3): pso_result_free()
//           - free(result->best_position); zero the struct.
//
// --- HOST DRIVER (new: main.cu or bench/main.cu) -----------------------------
//
// TODO(M3): CUDA_CHECK macro (wrap every cuda* / kernel launch).
// TODO(M3): main.cu — parse argv for {evaluator name, N, D, iters, seed},
//                     build PSOConfig, call pso_run, print best_value + pos.
// TODO(M3): cudaEvent timers around full run + each kernel; print ms breakdown.
// TODO(M3): Makefile / CMakeLists with -arch=sm_XX, link cuRAND (CUB header-only).
// TODO(M3): bench harness — write one CSV row per run with columns:
//   evaluator, N, D, iters, seed, eval_ms, reduce_ms, update_ms, total_ms,
//   final_gbest, achieved_bw_gbps, achieved_gflops
//   Append-only to bench/results.csv. Then sweeping is a shell for-loop and
//   the report's tables/figures come straight from the CSV (pandas/matplotlib).
//
// --- CPU BASELINE (new: cpu/pso_cpu.cpp) -------------------------------------
//
// TODO(M3): single-threaded reference PSO mirroring the GPU update equations.
// TODO(M3): host versions of Levy / Rastrigin / Schaffer F2 (mirror evals.cu).
// TODO(M3): correctness gate — GPU gbest matches CPU within tol on small problem.
// =============================================================================
