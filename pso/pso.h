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

//forward declare for MPI runs — full definition lives in mpi/mpi_island.h
//NB this is needed to declare the SyncCallback type above, which takes an IslandState pointer
typedef struct IslandState IslandState;

//sync callback type — called every sync_interval iterations in multi-island runs
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
    SyncCallback on_sync;
    void*        on_sync_data;
} PSOConfig;

//SofA format - coalescing. Downside - no swarm particle 'object'
typedef struct {
    float* positions;   // [n_particles * n_dims]
    float* velocities;  // [n_particles * n_dims]
    float* pbest_pos;   // [n_particles * n_dims]
    float* pbest;   // [n_particles] - overall best fitness of each particle
    float* fitness;     // [n_particles]
    //global best section
    float gbest_val; //scalar best fitness seen across all particles
    int gbest_idx; //global best index
    float* gbest_pos; //[n_dims] - needed for less wasteful memory management
    float* d_gbest_val; // device ptr, scalar best fitness
    int* d_gbest_idx; // device ptr, scalar best particle index

    //Reduction workspace — strategy-agnostic blob
    //(lets us call argmin() OR )
    void*  reduce_tmp;       // device ptr, opaque to caller
    size_t reduce_tmp_bytes; // size of above

    ReduceResult* d_reduce_out; // device ptr, single ReduceResult

    float* d_gbest_history; // device ptr, [max_iters], filled one entry/iter for convergence figure in progress report.
    // One curandState per particle. Per-iter randoms for the update kernel are
    // pregenerated into d_r1/d_r2 (N*D each) by kernel_draw_rng so the update
    // kernel itself does no RNG work and the per-iter RNG state traffic is N
    // instead of N*D.
    curandState* d_rng_states; // device ptr, [N]
    float* d_r1;               // device ptr, [N*D] — pregenerated each iter
    float* d_r2;               // device ptr, [N*D] — pregenerated each iter

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

// =============================================================================
// Milestone 3 — single-GPU TODOs (header surface)
// =============================================================================
// swarm_alloc(swarm*, const PSOConfig*)
//           cudaMalloc positions, velocities, pbest_pos, pbest, fitness,
//           gbest_pos, d_reduce_out, reduce_tmp workspace, cuRAND states.
// swarm_free(swarm*) — paired cudaFree, null out pointers.
// swarm_init(swarm*, const PSOConfig*)
//           launch curand_init kernel; fill positions ~ U[bound_lo, bound_hi];
//           velocities ~ U[-|hi-lo|, |hi-lo|] (or zero); seed pbest = +INF.
// =============================================================================
// Deferred to Milestone 4: n_islands / topology fields above stay unused.
// =============================================================================

cudaError_t swarm_alloc(swarm* s, const PSOConfig* cfg);
cudaError_t swarm_free(swarm* s);
cudaError_t swarm_init(swarm* s, const PSOConfig* cfg);
