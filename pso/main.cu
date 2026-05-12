// --- HOST DRIVER (new: main.cu or bench/main.cu) -----------------------------
//
// CUDA_CHECK macro (wrap every cuda* / kernel launch).
// TODO(M3): main.cu — parse argv for {evaluator name, N, D, iters, seed},
//                     build PSOConfig, call pso_run, print best_value + pos.
// TODO(M3): cudaEvent timers around full run + each kernel; print ms breakdown.
// TODO(M3): Makefile / CMakeLists with -arch=sm_XX, link cuRAND (CUB header-only).
// TODO(M3): bench harness — write one CSV row per run with columns:
//   evaluator, N, D, iters, seed, eval_ms, reduce_ms, update_ms, total_ms,
//   final_gbest, achieved_bw_gbps, achieved_gflops
//   Append-only to bench/results.csv. Then sweeping is a shell for-loop and
//   the report's tables/figures come straight from the CSV (pandas/matplotlib).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "evals.cuh"
#include "pso.h"

static void run_copy_gbest_pos_smoke_test(void) {
    constexpr int N = 4;
    constexpr int D = 3;
    constexpr int best_idx = 2;

    float h_pbest_pos[N * D];
    for (int d = 0; d < D; ++d) {
        for (int p = 0; p < N; ++p) {
            h_pbest_pos[d * N + p] = 100.0f * d + static_cast<float>(p);
        }
    }

    ReduceResult h_reduce{};
    h_reduce.val = -1.0f;
    h_reduce.idx = best_idx;

    float* d_pbest_pos = nullptr;
    float* d_gbest_pos = nullptr;
    ReduceResult* d_reduce = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pbest_pos, sizeof(h_pbest_pos)));
    CUDA_CHECK(cudaMalloc(&d_gbest_pos, sizeof(float) * D));
    CUDA_CHECK(cudaMalloc(&d_reduce, sizeof(ReduceResult)));
    CUDA_CHECK(cudaMemcpy(d_pbest_pos, h_pbest_pos, sizeof(h_pbest_pos), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reduce, &h_reduce, sizeof(ReduceResult), cudaMemcpyHostToDevice));

    kernel_copy_gbest_pos<<<1, D>>>(d_pbest_pos, d_gbest_pos, d_reduce, N, D);
    CUDA_CHECK(cudaGetLastError());

    float h_gbest_pos[D] = {};
    CUDA_CHECK(cudaMemcpy(h_gbest_pos, d_gbest_pos, sizeof(h_gbest_pos), cudaMemcpyDeviceToHost));

    for (int d = 0; d < D; ++d) {
        float expected = h_pbest_pos[d * N + best_idx];
        if (h_gbest_pos[d] != expected) {
            std::fprintf(stderr, "kernel_copy_gbest_pos smoke test failed at d=%d: got %g expected %g\n",
                d, h_gbest_pos[d], expected);
            std::exit(EXIT_FAILURE);
        }
    }

    CUDA_CHECK(cudaFree(d_pbest_pos));
    CUDA_CHECK(cudaFree(d_gbest_pos));
    CUDA_CHECK(cudaFree(d_reduce));
    std::printf("kernel_copy_gbest_pos smoke test passed.\n");
}

int main(void) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices visible; run this on a gpu-turing node.\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Using CUDA device 0: %s\n", prop.name);

    run_copy_gbest_pos_smoke_test();

    PSOConfig cfg = {
        .n_particles = 1024,
        .n_dims      = 30,
        .max_iters   = 100,
        .w           = 0.7f,
        .c1          = 1.5f,
        .c2          = 1.5f,
        .bound_lo    = -5.12f,
        .bound_hi    =  5.12f,
        .n_islands   = 1,
        .topology    = nullptr,
    };

    EvaluatorFn evaluator = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&evaluator, d_rastrigin_ptr, sizeof(EvaluatorFn)));
    PSOResult result = pso_run(&cfg, evaluator, 1, nullptr);

    std::printf("PSO run completed for Rastrigin N=%d D=%d iters=%d.\n",
        cfg.n_particles, cfg.n_dims, cfg.max_iters);
    std::printf("best_value = %.8g\n", result.best_value);
    std::printf("best_position[0] = %.6f\n", result.best_position[0]);

    pso_result_free(&result);
    return 0;
}
