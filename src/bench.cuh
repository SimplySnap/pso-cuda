//src/bench.cuh
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#include "../pso/pso.h"

/**
 * @brief One row of benchmark output, populated after pso_run() returns.
 *
 * @param impl             Implementation label ("gpu", "ring", "fc").
 * @param evaluator        Evaluator name string.
 * @param n_particles      Swarm size used.
 * @param n_dims           Problem dimensionality used.
 * @param max_iters        Iteration count used.
 * @param seed             Base RNG seed used.
 * @param eval_ms          CUDA-event time for eval stage (ms).
 * @param reduce_ms        CUDA-event time for reduce stage (ms).
 * @param update_ms        CUDA-event time for update stage (ms).
 * @param sync_ms          CUDA-event time for sync stage (ms); 0 for single-GPU.
 * @param total_ms         CUDA-event time for full run (ms).
 * @param final_gbest      Best objective value found.
 * @param achieved_bw_gbps Estimated memory bandwidth (GB/s).
 * @param achieved_gflops  Estimated throughput (GFLOP/s).
 */
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
    float              sync_ms;
    float              total_ms;
    double             final_gbest;
    double             achieved_bw_gbps;
    double             achieved_gflops;
};

static const char* kBenchCsvHeader =
    "impl,evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,sync_ms,"
    "total_ms,final_gbest,achieved_bw_gbps,achieved_gflops\n";

/**
 * @brief Appends one BenchRow to a CSV file, writing the header if the file
 *        is new or empty. Aborts on schema mismatch with an existing file.
 *
 * @param path Path to the output CSV file.
 * @param r    Populated BenchRow to append.
 *
 * @returns void. Calls exit(EXIT_FAILURE) on schema mismatch or open failure.
 *
 * @Structure
 *   - stat() to detect missing/empty file -> need_header
 *   - if file exists, fgets first line and strcmp against kBenchCsvHeader
 *   - fopen "a", conditionally fputs header, fprintf row
 */
static void append_bench_row(const char* path, const BenchRow& r) {
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
        "%s,%s,%d,%d,%d,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.8g,%.6f,%.6f\n",
        r.impl, r.evaluator,
        r.n_particles, r.n_dims, r.max_iters,
        (unsigned long long)r.seed,
        r.eval_ms, r.reduce_ms, r.update_ms, r.sync_ms, r.total_ms,
        r.final_gbest, r.achieved_bw_gbps, r.achieved_gflops);

    std::fclose(f);
}

/**
 * @brief Converts raw work and elapsed time into a throughput rate (Giga-units/s).
 *
 * @param work     Total units of work (bytes or flops).
 * @param total_ms Elapsed time in milliseconds.
 *
 * @returns Throughput in Giga-units/s, or 0.0 if total_ms <= 0.
 */
static double safe_rate(double work, float total_ms) {
    if (total_ms <= 0.0f) return 0.0;
    return work / (static_cast<double>(total_ms) * 1.0e6);
}

/**
 * @brief Analytical estimate of dominant global-memory traffic for the timed
 *        iteration loop. Intentionally skips CUB internals, cache effects,
 *        and cuRAND state traffic so the report formula stays readable.
 *
 * @param cfg PSO run configuration.
 *
 * @returns Estimated total bytes across all iterations.
 *
 * @Structure
 *   - eval:   read positions + write/update pbest_pos  -> 2 * N*D * float
 *   - reduce: read pbest values                        -> N * float
 *   - update: read x/v/pbest/gbest, write x/v         -> 5 * N*D * float
 */
static double estimate_loop_bytes(const PSOConfig& cfg) {
    const double N     = static_cast<double>(cfg.n_particles);
    const double D     = static_cast<double>(cfg.n_dims);
    const double I     = static_cast<double>(cfg.max_iters);
    const double fb    = static_cast<double>(sizeof(float));
    const double entry = N * D;

    const double eval_bytes   = entry * 2.0 * fb;
    const double reduce_bytes = N * fb;
    const double update_bytes = entry * 5.0 * fb;
    return I * (eval_bytes + reduce_bytes + update_bytes);
}

/**
 * @brief Coarse flop estimate for the timed iteration loop.
 *        Transcendentals are folded into the rough op count.
 *
 * @param cfg       PSO run configuration.
 * @param evaluator Evaluator name string.
 *
 * @returns Estimated total flops across all iterations.
 *
 * @Structure
 *   - rastrigin/levy: 10*D ops per particle
 *   - schaffer:       20 ops per particle (fixed 2D)
 *   - tsp:            D^2 ops per particle (argsort dominates)
 *   - update:         10 ops per particle per dim
 */
static double estimate_loop_flops(const PSOConfig& cfg, const char* evaluator) {
    const double N     = static_cast<double>(cfg.n_particles);
    const double D     = static_cast<double>(cfg.n_dims);
    const double I     = static_cast<double>(cfg.max_iters);
    const double entry = N * D;

    double eval_ops_per_particle = 10.0 * D;
    if      (std::strcmp(evaluator, "schaffer") == 0) eval_ops_per_particle = 20.0;
    else if (std::strcmp(evaluator, "tsp")      == 0) eval_ops_per_particle = D * D;

    const double eval_flops   = N * eval_ops_per_particle;
    const double update_flops = entry * 10.0;
    return I * (eval_flops + update_flops);
}