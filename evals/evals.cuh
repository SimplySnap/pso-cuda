#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math_constants.h>
#include "../pso/pso.h"

#ifndef PSO_EVALUATOR_FN_DEFINED
#define PSO_EVALUATOR_FN_DEFINED
typedef float (*EvaluatorFn)(const float* position, int n_dim);
#endif

//define pi
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// =============================================================================

extern __device__ EvaluatorFn d_levy_ptr;
extern __device__ EvaluatorFn d_rastrigin_ptr;
extern __device__ EvaluatorFn d_schaffer_ptr;
extern __device__ EvaluatorFn d_tsp_ptr;

// =============================================================================
// TSP (Traveling Salesman Problem) — a combinatorial benchmark.
// -----------------------------------------------------------------------------
// PSO optimizes continuous vectors, so TSP is encoded with "random keys" (a.k.a.
// Smallest-Position-Value, SPV): a particle's position x[D] holds one real key
// per city, and the tour is the permutation that sorts those keys ascending.
// Fitness = total Euclidean length of that closed tour. Only the *relative
// order* of the keys matters, so the existing update kernel and bounds work
// unchanged. Run with --evaluator tsp and D == number of cities.
//
// The instance (2D city coordinates) lives in __constant__ memory, uploaded
// once from the host via tsp_upload_instance() before pso_run(). MAX_TSP_CITIES
// bounds both the constant-memory footprint and the per-thread argsort buffer;
// it must not exceed the kernel's pos_local[] capacity (1024).
// =============================================================================
#ifndef MAX_TSP_CITIES
#define MAX_TSP_CITIES 1024
#endif

// Upload an instance of `n_cities` 2D points. `xy` is host memory of length
// 2*n_cities laid out [x0,y0, x1,y1, ...]. Returns a cudaError_t so the caller
// (which owns its own CUDA_CHECK macro) can validate. Fails if n_cities is out
// of (0, MAX_TSP_CITIES].
cudaError_t tsp_upload_instance(const float* xy, int n_cities);

// Number of cities currently uploaded (host-side mirror), or 0 if none.
int tsp_num_cities();
// Device function-pointer plumbing for EvaluatorFn
// -----------------------------------------------------------------------------
// pso_run() takes an EvaluatorFn (a __device__ function pointer), but the host
// CANNOT take the address of a __device__ function directly. The standard
// pattern is one __device__ pointer symbol per evaluator + cudaMemcpyFromSymbol
// on the host to materialize a launchable pointer:
//
typedef float (*EvaluatorFn)(const float*, int);
//
// Without this indirection, the kernel will silently dereference a host address
// and either segfault or return garbage fitness values.
// =============================================================================
