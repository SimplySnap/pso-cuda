
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

#include "pso.h"
#include "reduce.cuh"
#include "kernels.cuh"
#include "cuda_check.cuh"
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <limits>

cudaError_t swarm_alloc(swarm* s, const PSOConfig* cfg) {
    s->reduce_tmp_bytes = 0;
    CUDA_CHECK(cudaMalloc(&s->positions, sizeof(float) * cfg->n_particles * cfg->n_dims));
    CUDA_CHECK(cudaMalloc(&s->velocities, sizeof(float) * cfg->n_particles * cfg->n_dims));
    CUDA_CHECK(cudaMalloc(&s->pbest_pos, sizeof(float) * cfg->n_particles * cfg->n_dims));
    CUDA_CHECK(cudaMalloc(&s->pbest, sizeof(float) * cfg->n_particles));
    CUDA_CHECK(cudaMalloc(&s->fitness, sizeof(float) * cfg->n_particles));
    CUDA_CHECK(cudaMalloc(&s->gbest_pos, sizeof(float) * cfg->n_dims));
    CUDA_CHECK(cudaMalloc(&s->d_reduce_out, sizeof(ReduceResult)));
    CUDA_CHECK(cudaMalloc(&s->d_gbest_history, sizeof(float) * cfg->max_iters));
    //   Option 1- One state per thread in kernel_update (most flexible, max memory). 
    //   Let's go with Option 2:  One state per particle, with serial draws in update (less memory, less parallelism).
    //   This can be adjusted later if memory usage is a concern.
    int n_rng_states = cfg->n_particles * cfg->n_dims; // or cfg->n_particles if using one state per particle
    CUDA_CHECK(cudaMalloc(&s->d_rng_states, sizeof(curandState) * n_rng_states));

    // CUB reduction needs some temp storage for intermediate results. Query the size and allocate it here so we can reuse it across iterations.
    s->reduce_tmp_bytes = reduce_argmin_cub_workspace(cfg->n_particles);
    CUDA_CHECK(cudaMalloc(&s->reduce_tmp, s->reduce_tmp_bytes));

    return cudaSuccess;
}

cudaError_t swarm_free(swarm* s) {
    CUDA_CHECK(cudaFree(s->positions)); s->positions = nullptr;
    CUDA_CHECK(cudaFree(s->velocities)); s->velocities = nullptr;
    CUDA_CHECK(cudaFree(s->pbest_pos)); s->pbest_pos = nullptr; 
    CUDA_CHECK(cudaFree(s->pbest)); s->pbest = nullptr;
    CUDA_CHECK(cudaFree(s->fitness)); s->fitness = nullptr;
    CUDA_CHECK(cudaFree(s->gbest_pos)); s->gbest_pos = nullptr;
    CUDA_CHECK(cudaFree(s->d_reduce_out)); s->d_reduce_out = nullptr;
    CUDA_CHECK(cudaFree(s->d_gbest_history)); s->d_gbest_history = nullptr;
    CUDA_CHECK(cudaFree(s->d_rng_states)); s->d_rng_states = nullptr;
    CUDA_CHECK(cudaFree(s->reduce_tmp)); s->reduce_tmp = nullptr;
    return cudaSuccess;
}

cudaError_t swarm_init(swarm* s, const PSOConfig* cfg, unsigned long long seed) {
    // IMPORTANT: Ensure the first
    //            eval_and_pbest pass uses `<` against pbest=+INF (always true) and
    //            unconditionally copies positions->pbest_pos on that first hit.
    int n_rng_states = cfg->n_particles * cfg->n_dims; // or cfg->n_particles if using one state per particle
    dim3 block_rng(256);
    dim3 grid_rng((n_rng_states + block_rng.x - 1) / block_rng.x);
    kernel_curand_init<<<grid_rng, block_rng>>>(s->d_rng_states, seed, n_rng_states);
    CUDA_CHECK(cudaGetLastError());
    // Smoke-test/debug sync: catches runtime failures inside the RNG init kernel
    // at the exact lifecycle stage where they happen. Later PSO iterations should
    // avoid per-kernel synchronization unless timing or debugging requires it.
    CUDA_CHECK(cudaDeviceSynchronize());

    s->gbest_val = std::numeric_limits<float>::infinity();
    s->gbest_idx = -1;

    int n_entries = cfg->n_particles * cfg->n_dims;
    dim3 block_init(256);
    dim3 grid_init((n_entries + block_init.x - 1) / block_init.x);
    kernel_swarm_init<<<grid_init, block_init>>>(
        s->positions, s->velocities, s->pbest_pos, s->pbest, s->fitness,
        s->d_rng_states, cfg->bound_lo, cfg->bound_hi,
        cfg->n_particles, cfg->n_dims);
    CUDA_CHECK(cudaGetLastError());
    // Smoke-test/debug sync: makes swarm_init return only after device memory is
    // fully initialized, and surfaces any init-kernel error before main copies a sample.
    CUDA_CHECK(cudaDeviceSynchronize());
    return cudaSuccess;
}

__global__ void kernel_swarm_init(
    float*       __restrict__ positions,
    float*       __restrict__ velocities,
    float*       __restrict__ pbest_pos,
    float*       __restrict__ pbest,
    float*       __restrict__ fitness,
    curandState* __restrict__ d_states,
    float bound_lo, float bound_hi,
    int n_particles, int n_dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_entries = n_particles * n_dims;
    if (idx >= n_entries) return;

    // SoA layout: each dimension owns one contiguous stripe of N particles.
    // This makes positions[dim][particle] adjacent across particle IDs.
    int particle = idx % n_particles;
    int dim = idx / n_particles;
    int offset = dim * n_particles + particle;
    float span = bound_hi - bound_lo;

    // One RNG state per (particle, dim) entry. This matches the later planned
    // update kernel where each thread updates one position/velocity element.
    curandState local_state = d_states[idx];
    float rand_pos = bound_lo + curand_uniform(&local_state) * span;
    float rand_vel = (curand_uniform(&local_state) * 2.0f - 1.0f) * span;

    // Seed current position, velocity, and personal-best position consistently.
    // Fitness values stay +INF until the first evaluator kernel runs.
    positions[offset] = rand_pos;
    velocities[offset] = rand_vel;
    pbest_pos[offset] = rand_pos;

    // Only one dimension thread should initialize each particle-level scalar.
    if (dim == 0) {
        pbest[particle] = +CUDART_INF_F;
        fitness[particle] = +CUDART_INF_F;
    }

    // Save the advanced RNG state so future kernels continue the sequence.
    d_states[idx] = local_state;
}

PSOResult pso_run(const PSOConfig* cfg, EvaluatorFn evaluator, int islands, char* topology) {
    (void)islands;
    (void)topology;

    if (cfg == nullptr || evaluator == nullptr) {
        std::fprintf(stderr, "pso_run requires a non-null config and evaluator.\n");
        std::exit(EXIT_FAILURE);
    }

    swarm s{};
    CUDA_CHECK(swarm_alloc(&s, cfg));
    CUDA_CHECK(swarm_init(&s, cfg, 1234ULL));

    dim3 particle_block(256);
    dim3 particle_grid((cfg->n_particles + particle_block.x - 1) / particle_block.x);
    int n_entries = cfg->n_particles * cfg->n_dims;
    dim3 entry_block(256);
    dim3 entry_grid((n_entries + entry_block.x - 1) / entry_block.x);
    dim3 dim_block(256);
    dim3 dim_grid((cfg->n_dims + dim_block.x - 1) / dim_block.x);

    for (int iter = 0; iter < cfg->max_iters; ++iter) {
        kernel_eval_and_pbest<<<particle_grid, particle_block>>>(
            s.positions, s.fitness, s.pbest, s.pbest_pos,
            evaluator, cfg->n_particles, cfg->n_dims);
        CUDA_CHECK(cudaGetLastError());

        reduce_argmin_cub(
            s.pbest, cfg->n_particles,
            s.reduce_tmp, s.reduce_tmp_bytes,
            s.d_reduce_out, 0);

        ReduceResult h_reduce{};
        CUDA_CHECK(cudaMemcpy(&h_reduce, s.d_reduce_out,
            sizeof(ReduceResult), cudaMemcpyDeviceToHost));

        if (h_reduce.val < s.gbest_val) {
            s.gbest_val = h_reduce.val;
            s.gbest_idx = h_reduce.idx;
            kernel_copy_gbest_pos<<<dim_grid, dim_block>>>(
                s.pbest_pos, s.gbest_pos, s.d_reduce_out,
                cfg->n_particles, cfg->n_dims);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemcpy(s.d_gbest_history + iter, &s.gbest_val,
            sizeof(float), cudaMemcpyHostToDevice));

        if (s.gbest_idx >= 0) {
            kernel_update<<<entry_grid, entry_block>>>(
                s.positions, s.velocities, s.pbest_pos, s.gbest_pos,
                s.d_rng_states, cfg->w, cfg->c1, cfg->c2,
                cfg->bound_lo, cfg->bound_hi,
                cfg->n_particles, cfg->n_dims);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    PSOResult result{};
    result.best_position = static_cast<float*>(std::malloc(sizeof(float) * cfg->n_dims));
    if (result.best_position == nullptr) {
        std::fprintf(stderr, "Failed to allocate PSOResult.best_position.\n");
        CUDA_CHECK(swarm_free(&s));
        std::exit(EXIT_FAILURE);
    }

    result.best_value = s.gbest_val;
    if (s.gbest_idx >= 0) {
        CUDA_CHECK(cudaMemcpy(result.best_position, s.gbest_pos,
            sizeof(float) * cfg->n_dims, cudaMemcpyDeviceToHost));
    } else {
        for (int d = 0; d < cfg->n_dims; ++d) {
            result.best_position[d] = 0.0f;
        }
    }

    CUDA_CHECK(swarm_free(&s));
    return result;
}

void pso_result_free(PSOResult* result) {
    if (result == nullptr) return;

    std::free(result->best_position);
    result->best_position = nullptr;
    result->best_value = 0.0f;
}
