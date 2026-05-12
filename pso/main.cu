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

#include "cuda_check.cuh"
#include "evals.cuh"
#include "pso.h"

struct CliArgs {
    const char* evaluator;       // "rastrigin" | "levy" | "schaffer"
    int                n_particles;
    int                n_dims;
    int                max_iters;
    unsigned long long seed;
    const char*       csv_path;       // null = no CSV output
    const char*       history_path;   // null = don't dump gbest history
};

// One bench result. eval_ms / reduce_ms / update_ms come from per-kernel
// cudaEvent timers inside pso_run once that's wired up — zero for now.
struct BenchRow {
    const char*        evaluator;
    int                n_particles;
    int                n_dims;
    int                max_iters;
    unsigned long long seed;
    float              eval_ms;
    float              reduce_ms;
    float              update_ms;
    float              total_ms;
    double             final_gbest;
    double             achieved_bw_gbps;
    double             achieved_gflops;
};

static const char* kBenchCsvHeader =
    "evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,"
    "final_gbest,achieved_bw_gbps,achieved_gflops\n";

// Append one row to `path`. Writes the header iff the file doesn't exist yet,
// so concurrent shell sweeps stay valid CSV.
static void append_bench_row(const char* path, const BenchRow& r) {
    struct stat st{};
    bool need_header = (stat(path, &st) != 0);

    FILE* f = std::fopen(path, "a");
    if (!f) {
        std::fprintf(stderr, "could not open %s for append\n", path);
        return;
    }
    if (need_header) std::fputs(kBenchCsvHeader, f);

    std::fprintf(f,
        "%s,%d,%d,%d,%llu,%.4f,%.4f,%.4f,%.4f,%.8g,%.4f,%.4f\n",
        r.evaluator,
        r.n_particles, r.n_dims, r.max_iters,
        (unsigned long long)r.seed,
        r.eval_ms, r.reduce_ms, r.update_ms, r.total_ms,
        r.final_gbest, r.achieved_bw_gbps, r.achieved_gflops);

    std::fclose(f);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [--evaluator NAME] [--N INT] [--D INT] [--iters INT] [--seed UINT64]\n"
        "  evaluator: rastrigin (default) | levy | schaffer\n"
        "  N        : swarm size           (default 1024)\n"
        "  D        : dimensions           (default 30)\n"
        "  iters    : max iterations       (default 100)\n"
        "  seed     : RNG seed             (default 42)\n"
        "  csv_path : path to output CSV file (default none, i.e. no benchmarking output)\n"
        "  history  : write gbest-vs-iter history to this file (default: none)\n",
        prog);
}

// Resolve evaluator name -> device function pointer via cudaMemcpyFromSymbol.
// Returns nullptr if the name is unknown.
static EvaluatorFn resolve_evaluator(const char* name) {
    EvaluatorFn fn = nullptr;
    if (std::strcmp(name, "rastrigin") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_rastrigin_ptr, sizeof(fn)));
    } else if (std::strcmp(name, "levy") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_levy_ptr, sizeof(fn)));
    } else if (std::strcmp(name, "schaffer") == 0) {
        CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_schaffer_ptr, sizeof(fn)));
    }
    return fn;
}

static bool parse_args(int argc, char** argv, CliArgs* out) {
    // defaults
    out->evaluator   = "rastrigin";
    out->n_particles = 1024;
    out->n_dims      = 30;
    out->max_iters   = 100;
    out->seed        = 42ULL;
    out->csv_path     = nullptr;
    out->history_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: missing value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(a, "--evaluator") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->evaluator = v;
        } else if (std::strcmp(a, "--N") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->n_particles = std::atoi(v);
        } else if (std::strcmp(a, "--D") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->n_dims = std::atoi(v);
        } else if (std::strcmp(a, "--iters") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->max_iters = std::atoi(v);
        } else if (std::strcmp(a, "--seed") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->seed = std::strtoull(v, nullptr, 10);
        } else if (std::strcmp(a, "--csv_path") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->csv_path = v;
        } else if (std::strcmp(a, "--history") == 0) {
            const char* v = need_val(a); if (!v) return false;
            out->history_path = v;
        } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a);
            return false;
        }
    }

    if (out->n_particles <= 0 || out->n_dims <= 0 || out->max_iters <= 0) {
        std::fprintf(stderr, "N, D, iters must all be positive\n");
        return false;
    }
    return true;
}

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

int main(int argc, char** argv) {
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
        .n_particles = args.n_particles,
        .n_dims      = args.n_dims,
        .max_iters   = args.max_iters,
        .w           = 0.7f,
        .c1          = 1.5f,
        .c2          = 1.5f,
        .bound_lo    = -5.12f,
        .bound_hi    =  5.12f,
        .n_islands   = 1,
        .topology    = nullptr,
        .seed        = args.seed
        
    };

    EvaluatorFn evaluator = resolve_evaluator(args.evaluator);
    if (!evaluator) {
        std::fprintf(stderr, "unknown evaluator: %s\n", args.evaluator);
        return 1;
    }

    std::printf("running pso: evaluator=%s N=%d D=%d iters=%d seed=%llu\n",
        args.evaluator, cfg.n_particles, cfg.n_dims, cfg.max_iters,
        (unsigned long long)args.seed);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    PSOResult result = pso_run(&cfg, evaluator, 1, nullptr);

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    std::printf("best_value = %.8g\n", result.best_value);
    std::printf("best_position[0] = %.6f\n", result.best_position[0]);
    std::printf("total_ms = %.3f\n", total_ms);

    if (args.csv_path) {
        BenchRow row{};
        row.evaluator        = args.evaluator;
        row.n_particles      = cfg.n_particles;
        row.n_dims           = cfg.n_dims;
        row.max_iters        = cfg.max_iters;
        row.seed             = args.seed;
        row.eval_ms          = 0.0f;   // TODO(M3): fill from per-kernel timers in pso_run
        row.reduce_ms        = 0.0f;
        row.update_ms        = 0.0f;
        row.total_ms         = total_ms;
        row.final_gbest      = result.best_value;
        row.achieved_bw_gbps = 0.0;    // TODO(M3): compute from bytes_moved/total_ms
        row.achieved_gflops  = 0.0;    // TODO(M3): compute from flops/total_ms
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
