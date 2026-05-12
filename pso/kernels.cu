// --- KERNELS ----------------------
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

//   (b) Single-thread device kernel "kernel_commit_gbest" that reads
//       d_reduce_out, compares to a device-resident gbest_val, conditionally
//       updates it, and runs the copy_gbest_pos gather inline.
//       + Zero host sync in the inner loop; better for benchmarking.
//       - Need a separate path to dump gbest_history to host at end.
//   Recommend (b) + record gbest_history[it] from inside that same kernel.

#include "pso.h"
#include "kernels.cuh"
#include "reduce.cuh"
#include "reduce.cu"
#include "cuda_check.cuh"

__global__ void kernel_commit_gbest(
    const ReduceResult* __restrict__ d_reduce_out,
    const float*        __restrict__ pbest_pos,   // [N * D]
    float*                           gbest_pos,   // [D]
    float*                           d_gbest_val,
    int*                             d_gbest_idx,
    float*                           d_gbest_history,  // [max_iters]
    int   iter,
    int N, int D) {
        if (d_reduce_out->val < *d_gbest_val) {
            *d_gbest_val = d_reduce_out->val;
            *d_gbest_idx = d_reduce_out->idx;
            for (int d = 0; d < D; ++d) {
                gbest_pos[d] = pbest_pos[d*N + d_reduce_out->idx];
            }
        }
        d_gbest_history[iter] = *d_gbest_val;
}