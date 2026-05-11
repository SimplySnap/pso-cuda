// --- HOST DRIVER (new: main.cu or bench/main.cu) -----------------------------
//
// TODO(M3): CUDA_CHECK macro (wrap every cuda* / kernel launch).
// TODO(M3): main.cu — parse argv for {evaluator name, N, D, iters, seed},
//                     build PSOConfig, call pso_run, print best_value + pos.
// TODO(M3): cudaEvent timers around full run + each kernel; print ms breakdown.
// TODO(M3): Makefile / CMakeLists with -arch=sm_XX, link cuRAND (CUB header-only).
// TODO(M3): bench harness — write one CSV row per run with columns:
//   evaluator, N, D, iters, seed, eval_ms, reduce_ms, update_ms, total_ms,
//   final_gbest, achieved_bw_gbps, achieved_gflops
//   Append-only to bench/results.csv. Then sweeping is a shell for-loop and
//   the report's tables/figures come straight from the CSV (pandas/matplotlib).

// main.cu — entry point for pso-cuda.
// For now, just query and print CUDA device properties. Later, parse command-line
// args, call pso_run(), and print results.
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        return 1;                                                              \
    }                                                                          \
} while (0)

#include "pso.h"
#include "evals.cuh"

int main(void) {
    PSOConfig cfg = {
        .n_particles = 1024,
        .n_dims      = 30,
        .max_iters   = 500,
        .w           = 0.7f,
        .c1          = 1.5f,
        .c2          = 1.5f,
        .bound_lo    = -5.12f,
        .bound_hi    =  5.12f,
        .n_islands   = 1,
        .topology    = nullptr,
    };

    // EvaluatorFn h_fn;
    // cudaMemcpyFromSymbol(&h_fn, d_rastrigin_ptr, sizeof(EvaluatorFn));
    // PSOResult r = pso_run(&cfg, h_fn, /*islands=*/1);
    // printf("best = %g\n", r.best_value);
    // pso_result_free(&r);
    return 0;
}
