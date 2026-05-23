#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>

#include "../pso/cuda_check.cuh"
#include "../pso/pso.h"
#include "../evals/evals.cuh"
#include "../mpi/mpi_island.h"

//CliArgs, print_usage, parse_args, resolve_evaluator are identical to main_ring.cu.
//Shared in a future common header once benchmarks are running.

struct CliArgs {
    const char*        evaluator;
    int                n_particles;
    int                n_dims;
    int                max_iters;
    int                sync_interval;
    int                n_migrate;
    unsigned long long seed;
    const char*        csv_path;
    const char*        history_path;
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [--evaluator NAME] [--N INT] [--D INT] [--iters INT]\n"
        "          [--sync INT] [--migrate INT] [--seed UINT64]\n"
        "          [--csv_path PATH] [--history PATH]\n",
        prog);
}

static bool parse_args(int argc, char** argv, CliArgs* out) {
    /*
    Parses argv into CliArgs. Returns false on missing value or bad int.

    Args:
        argc (int):      Argument count.
        argv (char**):   Argument vector.
        out  (CliArgs*): Output struct.

    Returns:
        bool: true on success.
    */
    out->evaluator     = "rastrigin";
    out->n_particles   = 1024;
    out->n_dims        = 30;
    out->max_iters     = 100;
    out->sync_interval = 10;
    out->n_migrate     = 5;
    out->seed          = 42ULL;
    out->csv_path      = nullptr;
    out->history_path  = nullptr;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: missing value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if      (std::strcmp(a, "--evaluator") == 0) { const char* v = need_val(a); if (!v) return false; out->evaluator     = v; }
        else if (std::strcmp(a, "--N")         == 0) { const char* v = need_val(a); if (!v) return false; out->n_particles   = std::atoi(v); }
        else if (std::strcmp(a, "--D")         == 0) { const char* v = need_val(a); if (!v) return false; out->n_dims        = std::atoi(v); }
        else if (std::strcmp(a, "--iters")     == 0) { const char* v = need_val(a); if (!v) return false; out->max_iters     = std::atoi(v); }
        else if (std::strcmp(a, "--sync")      == 0) { const char* v = need_val(a); if (!v) return false; out->sync_interval = std::atoi(v); }
        else if (std::strcmp(a, "--migrate")   == 0) { const char* v = need_val(a); if (!v) return false; out->n_migrate     = std::atoi(v); }
        else if (std::strcmp(a, "--seed")      == 0) { const char* v = need_val(a); if (!v) return false; out->seed          = std::strtoull(v, nullptr, 10); }
        else if (std::strcmp(a, "--csv_path")  == 0) { const char* v = need_val(a); if (!v) return false; out->csv_path      = v; }
        else if (std::strcmp(a, "--history")   == 0) { const char* v = need_val(a); if (!v) return false; out->history_path  = v; }
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); return false; }
    }

    if (out->n_particles <= 0 || out->n_dims <= 0 || out->max_iters <= 0 || out->sync_interval <= 0) {
        std::fprintf(stderr, "N, D, iters, sync must all be positive\n");
        return false;
    }
    return true;
}

static EvaluatorFn resolve_evaluator(const char* name) {
    /*
    Resolves evaluator name to device function pointer via cudaMemcpyFromSymbol.

    Args:
        name (const char*): "rastrigin" | "levy" | "schaffer".

    Returns:
        EvaluatorFn: device function pointer, or nullptr if unknown.
    */
    EvaluatorFn fn = nullptr;
    if      (std::strcmp(name, "rastrigin") == 0) CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_rastrigin_ptr, sizeof(fn)));
    else if (std::strcmp(name, "levy")      == 0) CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_levy_ptr,      sizeof(fn)));
    else if (std::strcmp(name, "schaffer")  == 0) CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_schaffer_ptr,  sizeof(fn)));
    return fn;
}

int main(int argc, char** argv) {
    /*
    MPI fully-connected island PSO entry point.
    Each rank owns one island on one GPU. Every sync_interval iterations,
    all islands share their top n_migrate particles via MPI_Allgather,
    then exchange the global best position.

    Args:
        argc (int):    Argument count.
        argv (char**): Argument vector.

    Returns:
        int: 0 on success, 1 on error.

    Structure:
        - MPI_Init, rank/size
        - cudaSetDevice(rank % n_gpus)
        - parse_args, resolve_evaluator
        - island_sync_data_alloc
        - build PSOConfig with on_sync = island_migrate_fc
        - pso_run()
        - rank 0 collects and prints global best via MPI_Reduce
        - island_sync_data_free, MPI_Finalize
    */
    MPI_Init(&argc, &argv);

    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    CliArgs args;
    if (!parse_args(argc, argv, &args)) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (n_gpus == 0) {
        std::fprintf(stderr, "rank %d: no CUDA devices visible\n", rank);
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % n_gpus));

    if (rank == 0) {
        std::printf("pso_fc: %d islands, evaluator=%s N=%d D=%d iters=%d "
                    "sync_interval=%d migrate=%d seed=%llu\n",
                    n_ranks, args.evaluator, args.n_particles, args.n_dims,
                    args.max_iters, args.sync_interval, args.n_migrate,
                    (unsigned long long)args.seed);
    }

    EvaluatorFn evaluator = resolve_evaluator(args.evaluator);
    if (!evaluator) {
        std::fprintf(stderr, "rank %d: unknown evaluator: %s\n", rank, args.evaluator);
        MPI_Finalize();
        return 1;
    }

    IslandSyncData sync_data{};
    island_sync_data_alloc(&sync_data, MPI_COMM_WORLD, args.n_migrate, args.n_dims);

    PSOConfig cfg = {
        .n_particles   = args.n_particles,
        .n_dims        = args.n_dims,
        .max_iters     = args.max_iters,
        .w             = 0.7f,
        .c1            = 1.5f,
        .c2            = 1.5f,
        .bound_lo      = -5.12f,
        .bound_hi      =  5.12f,
        .n_islands     = n_ranks,
        .topology      = (char*)"fc",
        .seed          = args.seed + (unsigned long long)rank,
        .sync_interval = args.sync_interval,
        .on_sync       = island_migrate_fc,
        .on_sync_data  = &sync_data,
    };

    PSOResult result = pso_run(&cfg, evaluator, n_ranks, (char*)"fc");

    float local_val  = result.best_value;
    float global_val = 0.0f;
    MPI_Reduce(&local_val, &global_val, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("global best_value = %.8g\n", global_val);
        std::printf("eval_ms    = %.3f\n", result.eval_ms);
        std::printf("reduce_ms  = %.3f\n", result.reduce_ms);
        std::printf("update_ms  = %.3f\n", result.update_ms);
        std::printf("total_ms   = %.3f\n", result.total_ms);

        if (args.csv_path) {
            FILE* f = std::fopen(args.csv_path, "a");
            if (f) {
                std::fprintf(f,
                    "fc,%s,%d,%d,%d,%d,%llu,%.6f,%.6f,%.6f,%.6f,%.8g\n",
                    args.evaluator, n_ranks, args.n_particles, args.n_dims,
                    args.max_iters, (unsigned long long)args.seed,
                    result.eval_ms, result.reduce_ms, result.update_ms,
                    result.total_ms, global_val);
                std::fclose(f);
            }
        }
    }

    island_sync_data_free(&sync_data);
    pso_result_free(&result);
    MPI_Finalize();
    return 0;
}