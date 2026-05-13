#include "kernels.cuh"
#include "reduce.cuh"
#include "cuda_check.cuh"
#include <math_constants.h>

// --- KERNELS ----------------------
//
// __global__ kernel_curand_init(curandState* states, ull seed, int n)
//           One thread per RNG slot. Run once during swarm_init.
__global__ void kernel_curand_init(
    curandState* states,
    unsigned long long seed,
    int n_particles) {
    /*
    Initializes random seeds - runs once inside swarm_init(), before pso_run() ever starts
    cuRAND requires each state curandState to be initialized before curand_uniform()
    One thread per RNG slot to sample unique slots
    curand_uniform called exactly twice at the top of the thread/particle's body in update kernel
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    // Each particle gets a unique sequence number → non-overlapping
    // subsequences, even with a shared seed. 
    //offset=0 is fine here since we never fast-forward any state.
    curand_init(seed, /*sequence=*/tid, /*offset=*/0, &states[tid]);
}

//
// __global__ kernel_eval_and_pbest(
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
__global__ void kernel_eval_and_pbest(
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
    
    float pos_local[128]; //define local array
    //stage:
    for (int d = 0; d < D; d++)
        pos_local[d] = positions[tid + d * N];

    //b: eval fitness
    float fit = f(pos_local, D);
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
    int N, int D)
{
    /*
    One thread per particle. Each thread draws D pairs of (r1, r2) for its
    particle using a single curandState, then writes the state back. Pulling
    the state into a register and looping over D inside the thread keeps the
    per-iter RNG state traffic at N reads + N writes (×48B each) instead of
    N*D, which was dominating the bandwidth counter.
    */
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    curandState local = states[p];
    for (int d = 0; d < D; ++d) {
        int offset = d * N + p;
        r1[offset] = curand_uniform(&local);
        r2[offset] = curand_uniform(&local);
    }
    states[p] = local;
}
// __global__ kernel_update(
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
__global__ void kernel_update(
    float*       __restrict__ positions,
    float*       __restrict__ velocities,
    const float* __restrict__ pbest_pos,
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    const int*   __restrict__ d_gbest_idx,
    float w, float c1, float c2,
    float bound_lo, float bound_hi,
    int N, int D) {
    /*
    map thread ID to particle, block idx to dimension
    ensures coalesced loads! Specifically,
    SoA layout [d * N + particle], threads in the same warp have consecutive particle values and the same d
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_entries = N * D;
    if (idx >= n_entries) return; //guard

    int particle = idx % N;
    int dim = idx / N;
    int offset = dim * N + particle; //idx in memory

    // gbest_idx is broadcast across the warp; pbest_pos[dim*N + gidx] gives
    // the global-best particle's pbest for this dim (which IS the gbest pos).
    int gidx = *d_gbest_idx;
    float gb = pbest_pos[dim * N + gidx];

    float pos = positions[offset];
    float vel = velocities[offset];
    float pb = pbest_pos[offset];
    float r1v = r1[offset];
    float r2v = r2[offset];

    //update step
    float new_vel = w * vel
                  + c1 * r1v * (pb - pos)
                  + c2 * r2v * (gb - pos);

    float new_pos = pos + new_vel;

    //clamping
    if (new_pos < bound_lo) {
        new_pos = bound_lo;
        new_vel = 0.0f;
    } else if (new_pos > bound_hi) {
        new_pos = bound_hi;
        new_vel = 0.0f;
    }
    //write
    velocities[offset] = new_vel;
    positions[offset] = new_pos;
}

__global__ void kernel_commit_gbest(
    const ReduceResult* __restrict__ d_reduce_out,
    float*                           d_gbest_val,
    int*                             d_gbest_idx,
    float*                           d_gbest_history,  // [max_iters]
    int   iter,
    int N) {
        /*
        Scalar-only commit: read this iter's argmin, conditionally update
        d_gbest_val/d_gbest_idx, record d_gbest_history[iter]. The matching
        gbest_pos[D] gather is done once after the main loop — kernel_update
        reads pbest_pos[dim*N + *d_gbest_idx] directly so no per-iter D-wide
        copy is needed for correctness.
        */
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        int best_idx = d_reduce_out->idx;
        float best_val = d_reduce_out->val;
        if (best_idx >= 0 && best_idx < N && best_val < *d_gbest_val) {
            *d_gbest_val = best_val;
            *d_gbest_idx = best_idx;
        }
        d_gbest_history[iter] = *d_gbest_val;
}
