//#include <cstdio>
//#include <cstdlib>
//#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>

#include "../pso/cuda_check.cuh"
#include "../pso/pso.h"
#include "../evals/evals.cuh"
#include "../mpi/mpi_island.h"

#include "cli_args.cuh"
#include "bench.cuh"
#include "tsp_setup.cuh"

//CliArgs, print_usage, parse_args, resolve_evaluator are identical to main_ring.cu.
//Shared in a future common header once benchmarks are running.

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
    else if (std::strcmp(name, "tsp") == 0) CUDA_CHECK(cudaMemcpyFromSymbol(&fn, d_tsp_ptr, sizeof(fn)));
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

    //setup TSP
    if (std::strcmp(args.evaluator, "tsp") == 0) {
        if (setup_tsp_instance(args.tsp_file, args.n_dims, args.seed, &cfg) < 0) {
            MPI_Finalize(); return 1;
        }
    }

    IslandSyncData sync_data{};
    island_sync_data_alloc(&sync_data, MPI_COMM_WORLD, args.n_migrate, args.n_dims);

    PSOResult result = pso_run(&cfg, evaluator, n_ranks, (char*)"fc");

    float local_val  = result.best_value;
    float global_val = 0.0f;
    MPI_Reduce(&local_val, &global_val, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("global best_value = %.8g\n", global_val);
        std::printf("eval_ms    = %.3f\n", result.eval_ms);
        std::printf("reduce_ms  = %.3f\n", result.reduce_ms);
        std::printf("update_ms  = %.3f\n", result.update_ms);
        std::printf("sync_ms    = %.3f\n", result.sync_ms);
        std::printf("total_ms   = %.3f\n", result.total_ms);
        //standardize csv out
        if (args.csv_path) {
            BenchRow row{};
            row.impl             = "ring";   //or "fc" in main_fc.cu
            row.evaluator        = args.evaluator;
            row.n_particles      = cfg.n_particles;
            row.n_dims           = cfg.n_dims;
            row.max_iters        = cfg.max_iters;
            row.seed             = args.seed;
            row.eval_ms          = result.eval_ms;
            row.reduce_ms        = result.reduce_ms;
            row.update_ms        = result.update_ms;
            row.sync_ms          = result.sync_ms;
            row.total_ms         = result.total_ms;
            row.final_gbest      = global_val;
            row.achieved_bw_gbps = safe_rate(estimate_loop_bytes(cfg), result.total_ms);
            row.achieved_gflops  = safe_rate(estimate_loop_flops(cfg, args.evaluator), result.total_ms);
            append_bench_row(args.csv_path, row);
        }
    }

    island_sync_data_free(&sync_data);
    pso_result_free(&result);
    MPI_Finalize();
    return 0;
}