#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math_constants.h>

//define pi
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// =============================================================================
// TODO(M3): Device function-pointer plumbing for EvaluatorFn
// -----------------------------------------------------------------------------
// pso_run() takes an EvaluatorFn (a __device__ function pointer), but the host
// CANNOT take the address of a __device__ function directly. The standard
// pattern is one __device__ pointer symbol per evaluator + cudaMemcpyFromSymbol
// on the host to materialize a launchable pointer:
//
//   typedef float (*EvaluatorFn)(const float*, int);
//   __device__ EvaluatorFn d_levy_ptr        = levy_fn;
//   __device__ EvaluatorFn d_rastrigin_ptr   = rastrigin_fn;
//   __device__ EvaluatorFn d_schaffer_ptr    = schaffer_f2_fn;
//
//   // host:
//   EvaluatorFn h_fn;
//   cudaMemcpyFromSymbol(&h_fn, d_levy_ptr, sizeof(EvaluatorFn));
//   pso_run(&cfg, h_fn, /*islands=*/1);
//
// Without this indirection, the kernel will silently dereference a host address
// and either segfault or return garbage fitness values.
// =============================================================================