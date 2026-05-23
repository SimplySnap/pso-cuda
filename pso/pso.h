// include/pso/pso.h
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "reduce.cuh"

#ifndef PSO_EVALUATOR_FN_DEFINED
#define PSO_EVALUATOR_FN_DEFINED
// Device-callable evaluator function pointer type
typedef float (*EvaluatorFn)(const float* position, int n_dim);
#endif

/**
 * @brief Snapshot of one island's device-resident gbest state, passed to
 *        SyncCallback every sync_interval iterations.
 *
 * @param d_gbest_val  Device pointer to the scalar best fitness (float).
 * @param d_gbest_idx  Device pointer to the best particle index (int).
 * @param d_pbest_pos  Device pointer to pbest_pos[D*N] SoA array.
 * @param d_pbest_fit  Device pointer to pbest_fit[N] array.
 * @param N            Swarm size.
 * @param D            Number of dimensions.
 *
 * @Structure
 *   Plain data struct — no ownership. All device pointers are borrowed from
 *   the swarm struct inside pso_run() and are valid only during the callback.
 */
typedef struct {
    float* d_gbest_val;
    int*   d_gbest_idx;
    float* d_pbest_pos;
    float* d_pbest_fit;
    int    N;
    int    D;
} IslandState;

/**
 * @brief Callback invoked by pso_run() every sync_interval iterations.
 *
 * @param state      Pointer to the current island's device state snapshot.
 * @param user_data  Opaque pointer forwarded from PSOConfig.on_sync_data.
 *
 * @Structure
 *   The callback owns the synchronization logic (MPI exchange, migration, etc).
 *   It may read/write device memory via the pointers in state.
 *   pso_run() resumes immediately after the callback returns.
 */
typedef void (*SyncCallback)(IslandState* state, void* user_data);

/*Structures: config, best soln, particle*/
typedef struct {
    int   n_particles;   // swarm size
    int   n_dims;        // search space dimensionality
    int   max_iters;
    float w;             // inertia weight
    float c1;            // cognitive coefficient
    float c2;            // social coefficient
    float bound_lo;      // lower bound (per dim, uniform for now)
    float bound_hi;      // upper bound
    //now, for multi-island and multi-topology swarms:
    int n_islands;       // number of islands (and thus gpu clusters)
    char* topology;      // string topology
    unsigned long long seed; // RNG seed for reproducibility (optional, can be zero)

    //sync callback — null in single-GPU mode, set by MPI mains
    int          sync_interval; //call on_sync every this many iters (0 = disabled)
    SyncCallback on_sync;       //nullable
    void*        on_sync_data;  //forwarded to on_sync as user_data

} PSOConfig;

//SoA format — coalescing. Downside: no swarm particle 'object'
typedef struct {
    float* positions;
    float* velocities;
    float* pbest_pos;
    float* pbest;
    float* fitness;
    float  gbest_val;
    int    gbest_idx;
    float* gbest_pos;
    float* d_gbest_val;
    int*   d_gbest_idx;
    void*  reduce_tmp;
    size_t reduce_tmp_bytes;
    ReduceResult* d_reduce_out;
    float*        d_gbest_history;
    curandState*  d_rng_states;
    float*        d_r1;
    float*        d_r2;
} swarm;

typedef struct {
    float* best_position;  // host pointer, length n_dims
    float  best_value;
    float  eval_ms;
    float  reduce_ms;
    float  update_ms;
    float  total_ms;
    float* gbest_history;  // host pointer, length history_len (nullable)
    int    history_len;
} PSOResult;

// Main entry point — evaluator passed as parameter
PSOResult pso_run(const PSOConfig* cfg, EvaluatorFn evaluator, int islands, char* topology); 
//note islands is the number of independent swarms to run in parallel
//topology defines our communication topology
void pso_result_free(PSOResult* result);

cudaError_t swarm_alloc(swarm* s, const PSOConfig* cfg);
cudaError_t swarm_free(swarm* s);
cudaError_t swarm_init(swarm* s, const PSOConfig* cfg);
