/*
Kernel Declarations:
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

#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh> //holds high performance primitives (reduction argmin needed)
#include "../evals/evals.cuh"

// ── Reduction result ──────────────────────────────────────────────────────
/*
needs to return minimum value & its index. Lets us receive one pointer with everything needed for reduction
*/
typedef struct {
    float val;
    int   idx;
} ReduceResult;

//need to declare number of bytes for nvidia cub::DeviceReduce - cub doesn't allocate mem internally
//therefore, need to pass in buffer. This is what reduce_workspace_bytes gives us
size_t reduce_workspace_bytes(int n_particles);

//Kernel declarations

//two random numbers per iteration per particle
__global__ void kernel_curand_init(
    curandState* states,
    unsigned long long seed,
    int n);

// Eval + pbest update — one thread per particle
__global__ void kernel_eval_and_pbest(
    const float* __restrict__ positions,  // [N * D]
    float*       __restrict__ fitness,    // [N] - scalar each
    float*       __restrict__ pbest_fit,  // [N] - pbest scalar fitness
    float*       __restrict__ pbest_pos,  // [N * D]
    int N, int D);

// Argmin reduction — wraps CUB; swap body for custom later
void reduce_argmin(
    const float*  __restrict__ d_fitness,
    ReduceResult* d_out,
    void*         tmp,
    size_t        tmp_bytes,
    int           n_particles,
    cudaStream_t  stream);

// Commit gbest — device-resident compare + conditional copy
__global__ void kernel_commit_gbest(
    const ReduceResult* __restrict__ d_reduce_out,
    const float*        __restrict__ pbest_pos,   // [N * D]
    float*                           gbest_pos,   // [D]
    float*                           d_gbest_val,
    int*                             d_gbest_idx,
    float*                           d_gbest_history,  // [max_iters]
    int   iter,
    int N, int D);

    // 5. Velocity + position update — one thread per (particle × dim)
__global__ void kernel_update(
    float*       __restrict__ positions,   // [N * D]
    float*       __restrict__ velocities,  // [N * D]
    const float* __restrict__ pbest_pos,   // [N * D]
    const float* __restrict__ gbest_pos,   // [D]
    curandState* __restrict__ rng_states,
    float w, float c1, float c2,
    float bound_lo, float bound_hi,
    int N, int D);
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