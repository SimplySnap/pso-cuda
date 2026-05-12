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

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "pso.h"

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

    swarm s{};
    CUDA_CHECK(swarm_alloc(&s, &cfg));
    CUDA_CHECK(swarm_init(&s, &cfg, 1234ULL));

    float first_position = 0.0f;
    float first_pbest = 0.0f;
    CUDA_CHECK(cudaMemcpy(&first_position, s.positions, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&first_pbest, s.pbest, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Smoke sample: positions[dim=0, particle=0] = %.6f, pbest[0] = %g\n",
        first_position, first_pbest);
    std::printf("Host gbest initialized: value = %g, idx = %d\n", s.gbest_val, s.gbest_idx);

    CUDA_CHECK(swarm_free(&s));
    // Smoke-test/debug sync: confirms all CUDA work, including cleanup-adjacent
    // runtime bookkeeping, has completed before reporting success.
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("PSO swarm alloc/init/free smoke test passed for N=%d, D=%d.\n",
        cfg.n_particles, cfg.n_dims);
    return 0;
}
