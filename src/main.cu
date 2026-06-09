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
#include <cstring>
#include <cuda_runtime.h>
#include <sys/stat.h> 

#include "../pso/cuda_check.cuh" //new file structure
#include "../evals/evals.cuh"
#include "../pso/pso.h"
#include "tsp_setup.cuh"
#include "cli_args.cuh"
#include "bench.cuh"


static EvaluatorFn resolve_evaluator(const char* name) {
    /*
    Resolves an evaluator name string to a device function pointer via cudaMemcpyFromSymbol.

    Args:
        name (const char*): One of "rastrigin", "levy", "schaffer".

    Returns:
        EvaluatorFn: Device-callable function pointer, or nullptr if name is unknown.
    */
    EvaluatorFn fn = nullptr;
    if (std::strcmp(name, "rastrigin") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_rastrigin_ptr, sizeof(fn)));
    } else if (std::strcmp(name, "levy") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_levy_ptr, sizeof(fn)));
    } else if (std::strcmp(name, "schaffer") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_schaffer_ptr, sizeof(fn)));
    } else if (std::strcmp(name, "tsp") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_tsp_ptr, sizeof(fn)));
    }
    return fn;
}

static void run_copy_gbest_pos_smoke_test(void) {
    /*
    Smoke test for kernel_copy_gbest_pos. Verifies that the kernel correctly
    gathers the winning particle's position into gbest_pos for a known best_idx.

    Args: none
    Returns: void. Calls exit(EXIT_FAILURE) on any mismatch.

    Structure:
        - Build small N=4, D=3 pbest_pos array on host with known pattern
        - Copy to device, set best_idx=2, launch kernel, copy result back
        - Assert gbest_pos[d] == pbest_pos[d*N + best_idx] for all d
    */
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

//MPI - nodes use Allreduce with 'max' of gbest_val to periodically sync gbest

int main(int argc, char** argv) {
    /*
    Single-GPU PSO entry point. Parses CLI args, runs pso_run(), prints results,
    optionally writes bench CSV and gbest history file.

    Args:
        argc (int):    Argument count.
        argv (char**): Argument vector.

    Returns:
        int: 0 on success, 1 on error.

    Structure:
        - parse_args -> CliArgs
        - cudaSetDevice(0), print device name
        - run_copy_gbest_pos_smoke_test()
        - build PSOConfig, resolve_evaluator, pso_run()
        - print results, optionally append_bench_row, dump history
        - pso_result_free()
    */
    CliArgs args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

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
        .n_particles   = args.n_particles,
        .n_dims        = args.n_dims,
        .max_iters     = args.max_iters,
        .w             = 0.7f,
        .c1            = 1.5f,
        .c2            = 1.5f,
        .bound_lo      = -5.12f,
        .bound_hi      =  5.12f,
        .n_islands     = 1,
        .topology      = nullptr,
        .seed          = args.seed,
        .sync_interval = 0,
        .on_sync       = nullptr,
        .on_sync_data  = nullptr,
    };

    EvaluatorFn evaluator = resolve_evaluator(args.evaluator);
    if (!evaluator) {
        std::fprintf(stderr, "unknown evaluator: %s\n", args.evaluator);
        return 1;
    }

    // TSP needs its instance uploaded to constant memory; this also pins
    // cfg.n_dims = #cities and sets random-keys bounds [0,1].
    if (std::strcmp(args.evaluator, "tsp") == 0) {
        if (setup_tsp_instance(args.tsp_file, args.n_dims, args.seed, &cfg) < 0) return 1;
    }

    std::printf("running pso: evaluator=%s N=%d D=%d iters=%d seed=%llu\n",
        args.evaluator, cfg.n_particles, cfg.n_dims, cfg.max_iters,
        (unsigned long long)args.seed);

    PSOResult result = pso_run(&cfg, evaluator, 1, nullptr);

    std::printf("best_value = %.8g\n", result.best_value);
    std::printf("best_position[0] = %.6f\n", result.best_position[0]);
    std::printf("eval_ms = %.3f\n", result.eval_ms);
    std::printf("reduce_ms = %.3f\n", result.reduce_ms);
    std::printf("update_ms = %.3f\n", result.update_ms);
    std::printf("total_ms = %.3f\n", result.total_ms);

    if (args.csv_path) {
        BenchRow row{};
        row.impl             = "gpu";
        row.evaluator        = args.evaluator;
        row.n_particles      = cfg.n_particles;
        row.n_dims           = cfg.n_dims;
        row.max_iters        = cfg.max_iters;
        row.seed             = args.seed;
        row.w  = cfg.w;
        row.c1 = cfg.c1;
        row.c2 = cfg.c2;
        row.eval_ms          = result.eval_ms;
        row.reduce_ms        = result.reduce_ms;
        row.update_ms        = result.update_ms;
        row.sync_ms          = 0.0f; //single GPU - no sync stage
        row.total_ms         = result.total_ms;
        row.final_gbest      = result.best_value;
        row.achieved_bw_gbps = safe_rate(estimate_loop_bytes(cfg), result.total_ms);
        row.achieved_gflops  = safe_rate(estimate_loop_flops(cfg, args.evaluator), result.total_ms);
        append_bench_row(args.csv_path, row);
    }

    // gbest-vs-iter dump
    if (result.gbest_history != nullptr && result.history_len > 0) {
        // Sampled stdout summary — at most ~20 lines so it's readable for any iters.
        int stride = result.history_len > 20 ? result.history_len / 20 : 1;
        std::printf("\ngbest history (sampled every %d iters):\n", stride);
        for (int i = 0; i < result.history_len; i += stride) {
            std::printf("  iter %5d: %.6g\n", i, result.gbest_history[i]);
        }
        std::printf("  iter %5d: %.6g  (final)\n",
            result.history_len - 1,
            result.gbest_history[result.history_len - 1]);

        // Full per-iter dump to file, two columns: iter,gbest
        if (args.history_path) {
            FILE* hf = std::fopen(args.history_path, "w");
            if (!hf) {
                std::fprintf(stderr, "could not open %s for write\n", args.history_path);
            } else {
                std::fputs("iter,gbest\n", hf);
                for (int i = 0; i < result.history_len; ++i) {
                    std::fprintf(hf, "%d,%.8g\n", i, result.gbest_history[i]);
                }
                std::fclose(hf);
                std::printf("wrote full history to %s\n", args.history_path);
            }
        }
    }

    std::printf("PSO run completed for %s N=%d D=%d iters=%d seed=%llu.\n",
        args.evaluator, cfg.n_particles, cfg.n_dims, cfg.max_iters, (unsigned long long)cfg.seed);
    pso_result_free(&result);
    return 0;
}
