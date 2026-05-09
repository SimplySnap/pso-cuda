// include/pso/pso.h
#pragma once
#include <stddef.h> 
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// Device-callable evaluator function pointer type
typedef float (*EvaluatorFn)(const float* position, int n_dim);

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
} swarm;

typedef struct {
    float* best_position;  // host pointer, length n_dims
    float  best_value;
} PSOResult;

// Main entry point — evaluator passed as parameter
PSOResult pso_run(const PSOConfig* cfg, EvaluatorFn evaluator, int islands); //note islands is the number of independent swarms to run in parallel - defines our communication topology
void pso_result_free(PSOResult* result);