//src/cli_args.cuh
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * @brief Parsed CLI arguments, shared across all PSO entry points.
 *        MPI mains use sync_interval, n_migrate, and n_islands;
 *        single-GPU main leaves them at their defaults.
 *
 * @param evaluator     Evaluator name: "rastrigin"|"levy"|"schaffer"|"tsp".
 * @param n_particles   Swarm size per island.
 * @param n_dims        Problem dimensionality (or city count for tsp).
 * @param max_iters     Maximum PSO iterations.
 * @param sync_interval Iterations between MPI sync callbacks (MPI mains only).
 * @param n_migrate     Particles to migrate per sync (MPI mains only).
 * @param seed          Base RNG seed; MPI mains add rank offset for PSO only.
 * @param csv_path      Path for benchmark CSV output; nullptr = no output.
 * @param history_path  Path for gbest-vs-iter dump; nullptr = no output.
 * @param tsp_file      Path to "x y" TSP instance file; nullptr = random.
 */
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
    const char*        tsp_file;
};

/**
 * @brief Prints usage to stderr.
 *
 * @param prog argv[0].
 *
 * @returns void.
 */
static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [--evaluator NAME] [--N INT] [--D INT] [--iters INT]\n"
        "          [--sync INT] [--migrate INT] [--seed UINT64]\n"
        "          [--csv_path PATH] [--history PATH] [--tsp_file PATH]\n"
        "  evaluator : rastrigin (default) | levy | schaffer | tsp\n"
        "  N         : particles per island  (default 1024)\n"
        "  D         : dimensions            (default 30)\n"
        "  iters     : max iterations        (default 100)\n"
        "  sync      : sync interval (MPI)   (default 10)\n"
        "  migrate   : particles to migrate  (default 5)\n"
        "  seed      : base RNG seed         (default 42)\n"
        "  csv_path  : benchmark CSV output  (default none)\n"
        "  history   : gbest-vs-iter file    (default none)\n"
        "  tsp_file  : tsp instance file     (default random from seed)\n",
        prog);
}

/**
 * @brief Parses argv into a CliArgs struct with sensible defaults.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param out  Output CliArgs to populate.
 *
 * @returns true on success, false on missing value or invalid integer.
 *
 * @Structure
 *   - set defaults
 *   - iterate argv, match flags, call need_val lambda for values
 *   - validate n_particles, n_dims, max_iters > 0
 */
static bool parse_args(int argc, char** argv, CliArgs* out) {
    out->evaluator     = "rastrigin";
    out->n_particles   = 1024;
    out->n_dims        = 30;
    out->max_iters     = 100;
    out->sync_interval = 10;
    out->n_migrate     = 5;
    out->seed          = 42ULL;
    out->csv_path      = nullptr;
    out->history_path  = nullptr;
    out->tsp_file      = nullptr;

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
        else if (std::strcmp(a, "--tsp_file")  == 0) { const char* v = need_val(a); if (!v) return false; out->tsp_file      = v; }
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", a); return false; }
    }

    if (out->n_particles <= 0 || out->n_dims <= 0 || out->max_iters <= 0) {
        std::fprintf(stderr, "N, D, iters must all be positive\n");
        return false;
    }
    return true;
}