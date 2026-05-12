// --- KERNELS ----------------------
//
// TODO(M3): __global__ kernel_curand_init(curandState* states, ull seed, int n)
//           One thread per RNG slot. Run once during swarm_init.
__global__ kernel_curand_init(curandState* states, ull seed, int N){
    /*
    Initializes random seeds - runs once inside swarm_init(), before pso_run() ever starts
    cuRAND requires each state curandState to be initialized before curand_uniform()
    One thread per RNG slot to sample unique slots
    curand_uniform called exactly twice at the top of the thread/particle's body in update kernel
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Each particle gets a unique sequence number → non-overlapping
    // subsequences, even with a shared seed. 
    //offset=0 is fine here since we never fast-forward any state.
    curand_init(seed, /*sequence=*/tid, /*offset=*/0, &states[tid]);
}

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
__global__ kernel_eval_and_pbest(
    const float* positions, //[D*N]
    float* fitness, //[N]
    float*  pbest_fit, //[N]
    float* pbest_pos, //[D*N]
    EvaluatorFn f,
    int N, int D
){
    /*
    Assigns thread particle, and checks if fitness[i]<pbest_fit[i]
    If so, update pbest_pos[i] and pbest_fit[i]
    */
   //index
   int tid = blockDim.x*blockIdx.x+threadIdx.x;
   if (tid >= N) return;

    //1: evaluate fitness for current particle
    //a: stage fitness array to pass into f as _contiguous_ array
    
    float pos_local[MAX_D]; //define local array
    //stage:
    for (int d = 0; d < D; d++)
        pos_local[d] = positions[tid + d * N];

    //b: eval fitness
    float fit = f(&pos_local, D);
    fitness[tid]=fit;

    //2: conditionally update
    if (fit<pbest_fit[tid]){
        //a:overwrite
        pbest_fit[tid]=fit;

        //b: particle D coords to pbest_pos
        for (int d = 0; d < D; ++d) {
            pbest_pos[tid + d * N] = positions[tid + d * N];
        }
   }
}

__global__ void kernel_draw_rng(
    curandState* __restrict__ states,
    float*       __restrict__ r1,
    float*       __restrict__ r2,
    int N)
{
    /*
    NEEDED FOR UPDATE - pregenerate random vectors (r1 holds N random numbers)
    pass in to update kernel to avoid race conditions to rng
    */
    int part = blockIdx.x * blockDim.x + threadIdx.x;
    if (part >= N) return;
    r1[part] = curand_uniform(&states[part]);
    r2[part] = curand_uniform(&states[part]);
}
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
__global__ kernel_update(
    float*       __restrict__ positions, float* __restrict__ velocities,
    const float* __restrict__ pbest_pos, const float* __restrict__ gbest_pos,
    const float* __restrict__ r1,        // [N] — pre-drawn
    const float* __restrict__ r2,        // [N] — pre-drawn
    float w, float c1, float c2,
    float bound_lo, float bound_hi,
    int N, int D
){
    /*
    map thread ID to particle, block idx to dimension
    ensures coalesced loads! Specifically,
    SoA layout [d * N + particle], threads in the same warp have consecutive particle values and the same d
    */
    int part = blockIdx.x * blockDim.x + threadIdx.x;  // which particle
    int dim = blockIdx.y; // which dimension

    if (particle >= N || d >= D) return; //guard

    int idx = dim*N + part; //actual index

    // r1, r2 drawn once per particle — only dim==0 thread advances the state,
    // then broadcasts to all dims of this particle via shared memory
    // Simple alternative: draw per (particle, dim) — statistically fine for PSO
    // and avoids cross-dim coordination. We use that here:

    float r1_val = r1[part];   // same value for all dims of this particle
    float r2_val = r2[part];

    // Load current state
    float pos  = positions[idx];
    float vel  = velocities[idx];
    float pb   = pbest_pos[idx];
    float gb   = gbest_pos[dim];   // broadcast: same for all particles at this dim

    // PSO velocity update
    float new_vel = w  * vel
                  + c1 * r1 * (pb - pos)
                  + c2 * r2 * (gb - pos);

    // Position update + clamp to bounds
    float new_pos = pos + new_vel;
    new_pos = fmaxf(bound_lo, fminf(bound_hi, new_pos));

    velocities[idx] = new_vel;
    positions[idx]  = new_pos;
}
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