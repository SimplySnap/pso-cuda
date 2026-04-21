// include/pso/pso.h
#pragma once
#include <stddef.h> 

// Device-callable evaluator function pointer type
typedef float (*EvaluatorFn)(const float* position, int dims);

typedef struct {
    int   n_particles;   // swarm size
    int   n_dims;        // search space dimensionality
    int   max_iters;
    float w;             // inertia weight
    float c1;            // cognitive coefficient
    float c2;            // social coefficient
    float bound_lo;      // lower bound (per dim, uniform for now)
    float bound_hi;      // upper bound
} PSOConfig;

typedef struct {
    float* best_position;  // host pointer, length n_dims
    float  best_value;
} PSOResult;

// Main entry point — evaluator passed as parameter
PSOResult pso_run(const PSOConfig* cfg, EvaluatorFn evaluator, int islands); //note islands is the number of independent swarms to run in parallel - defines our communication topology
void pso_result_free(PSOResult* result);