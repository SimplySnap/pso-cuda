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
