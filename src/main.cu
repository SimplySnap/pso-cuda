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

struct CliArgs {
    const char* evaluator;       // "rastrigin" | "levy" | "schaffer" etc
    int                n_particles;
    int                n_dims;
    int                max_iters;
    unsigned long long seed;
    const char*       csv_path;       // null = no CSV output
    const char*       history_path;   // null = don't dump gbest history
};

// One bench result. Timing fields are loop-stage CUDA event timings from pso_run.
struct BenchRow {
    const char*        impl;
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
    "impl,evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,"
    "final_gbest,achieved_bw_gbps,achieved_gflops\n";

static void append_bench_row(const char* path, const BenchRow& r) {
    /*
    Appends one benchmark result row to a CSV file.
    Writes the header if the file is new or empty, and aborts on schema mismatch.

    Args:
        path (const char*): Path to the output CSV file.
        r    (const BenchRow&): Populated benchmark result struct.

    Returns:
        void. Calls exit(EXIT_FAILURE) on schema mismatch or file open failure.

    Structure:
        - stat() to detect missing/empty file -> need_header
        - If file exists, fgets first line and strcmp against kBenchCsvHeader
        - fopen "a", conditionally fputs header, fprintf row
    */
    struct stat st{};
    bool need_header = (stat(path, &st) != 0 || st.st_size == 0);

    if (!need_header) {
        FILE* existing = std::fopen(path, "r");
        if (!existing) {
            std::fprintf(stderr, "could not open %s for header validation\n", path);
            std::exit(EXIT_FAILURE);
        }

        char header[512] = {};
        if (std::fgets(header, sizeof(header), existing) == nullptr) {
            need_header = true;
        } else if (std::strcmp(header, kBenchCsvHeader) != 0) {
            std::fprintf(stderr,
                "CSV header mismatch in %s; refusing to append mixed-schema rows.\n"
                "expected: %s"
                "found:    %s",
                path, kBenchCsvHeader, header);
            std::fclose(existing);
            std::exit(EXIT_FAILURE);
        }
        std::fclose(existing);
    }

    FILE* f = std::fopen(path, "a");
    if (!f) {
        std::fprintf(stderr, "could not open %s for append\n", path);
        return;
    }
    if (need_header) std::fputs(kBenchCsvHeader, f);

    std::fprintf(f,
        "%s,%s,%d,%d,%d,%llu,%.6f,%.6f,%.6f,%.6f,%.8g,%.6f,%.6f\n",
        r.impl, r.evaluator,
        r.n_particles, r.n_dims, r.max_iters,
        (unsigned long long)r.seed,
        r.eval_ms, r.reduce_ms, r.update_ms, r.total_ms,
        r.final_gbest, r.achieved_bw_gbps, r.achieved_gflops);

    std::fclose(f);
}

static double safe_rate(double work, float total_ms) {
    /*
    Converts raw work and elapsed time into a throughput rate (Giga-units/s).

    Args:
        work     (double): Total units of work (bytes or flops).
        total_ms (float):  Elapsed time in milliseconds.

    Returns:
        double: Throughput in Giga-units per second, or 0.0 if total_ms <= 0.
    */
    if (total_ms <= 0.0f) return 0.0;
    return work / (static_cast<double>(total_ms) * 1.0e6);
}

// Simple analytical loop-traffic estimate for M3 tables, not an Nsight
// hardware-counter measurement. These bytes are meant to explain the dominant
// global-memory traffic in the timed iteration loop:
//   - eval: read positions and write/update pbest_pos
//   - reduce: read pbest values
//   - update: read positions/velocities/pbest/gbest and write positions/velocities
// This intentionally skips CUB internals, cache effects, exact branch-dependent
// pbest writes, and cuRAND state traffic so the report formula stays readable.
static double estimate_loop_bytes(const PSOConfig& cfg) {
    /*
    Analytical estimate of dominant global-memory traffic for the timed iteration loop.
    Intentionally skips CUB internals, cache effects, and cuRAND state traffic.

    Args:
        cfg (const PSOConfig&): PSO run configuration.

    Returns:
        double: Estimated total bytes across all iterations.

    Structure:
        - eval:   read positions + write/update pbest_pos  -> 2 * N*D * float
        - reduce: read pbest values                        -> N * float
        - update: read x/v/pbest/gbest, write x/v         -> 5 * N*D * float
    */
    const double N = static_cast<double>(cfg.n_particles);
    const double D = static_cast<double>(cfg.n_dims);
    const double I = static_cast<double>(cfg.max_iters);
    const double entries = N * D;
    const double float_b = static_cast<double>(sizeof(float));

    const double eval_bytes = entries * 2.0 * float_b;    // read positions, write/update pbest_pos
    const double reduce_bytes = N * float_b;              // read pbest values
    const double update_bytes = entries * 5.0 * float_b;  // read x/v/pbest/gbest, write x/v
    return I * (eval_bytes + reduce_bytes + update_bytes);
}

static double estimate_loop_flops(const PSOConfig& cfg, const char* evaluator) {
    /*
    Coarse flop estimate for the timed iteration loop.
    Transcendental functions are counted as part of the rough op count.

    Args:
        cfg       (const PSOConfig&): PSO run configuration.
        evaluator (const char*):      Evaluator name string.

    Returns:
        double: Estimated total flops across all iterations.
    */
    const double N = static_cast<double>(cfg.n_particles);
    const double D = static_cast<double>(cfg.n_dims);
    const double I = static_cast<double>(cfg.max_iters);
    const double entries = N * D;

    // Coarse compute estimate for the timed iteration loop. eval_flops estimates
    // the objective-function math inside evals.cu, which is called by
    // kernel_eval_and_pbest; it does not try to count the surrounding pbest
    // compare/copy bookkeeping. update_flops estimates the arithmetic in
    // kernel_update's velocity/position equation.
    //
    // Rastrigin and Levy scale with D, so use a simple 10*D ops per particle.
    // Schaffer F2 is fixed 2D, so use a small constant. These constants are
    // deliberately rough and treat transcendental functions as part of the
    // coarse operation count rather than a precise instruction count.
    double eval_ops_per_particle = 10.0 * D;
    if (std::strcmp(evaluator, "schaffer") == 0) {
        eval_ops_per_particle = 20.0;
    }

    const double eval_flops = N * eval_ops_per_particle;
    const double update_flops = entries * 10.0;
    return I * (eval_flops + update_flops);
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
    }
    return fn;
}

static bool parse_args(int argc, char** argv, CliArgs* out) {
    /*
    Parses argv into a CliArgs struct. Exits on bad input.

    Args:
        argc (int):      Argument count.
        argv (char**):   Argument vector.
        out  (CliArgs*): Output struct to populate.

    Returns:
        bool: true on success, false if a required value is missing or invalid.
    */
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
        .sync_interval = 0,       //disabled in single-GPU mode
        .on_sync       = nullptr,
        .on_sync_data  = nullptr,
    };

    EvaluatorFn evaluator = resolve_evaluator(args.evaluator);
    if (!evaluator) {
        std::fprintf(stderr, "unknown evaluator: %s\n", args.evaluator);
        return 1;
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
        row.eval_ms          = result.eval_ms;
        row.reduce_ms        = result.reduce_ms;
        row.update_ms        = result.update_ms;
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
