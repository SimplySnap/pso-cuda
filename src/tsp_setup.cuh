//src/tsp_setup.cuh
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../evals/evals.cuh"
#include "../pso/pso.h"

/**
 * @brief Reads a TSP instance from a file of "x y" newline-separated pairs
 *        into a flat host buffer [x0,y0,x1,y1,...].
 *
 * @param path       Path to the instance file.
 * @param xy         Output buffer; caller must allocate at least
 *                   2 * max_cities * sizeof(float) bytes.
 * @param max_cities Maximum cities to read (bounded by MAX_TSP_CITIES).
 *
 * @returns City count on success, or -1 on file error / fewer than 2 cities.
 *
 * @Structure
 *   - fopen, fscanf loop up to max_cities pairs
 *   - fclose
 *   - return -1 if n < 2
 */
static int tsp_load_file(const char* path, float* xy, int max_cities) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "tsp: could not open file %s\n", path);
        return -1;
    }
    int n = 0;
    float x, y;
    while (n < max_cities && std::fscanf(f, "%f %f", &x, &y) == 2) {
        xy[2 * n]     = x;
        xy[2 * n + 1] = y;
        ++n;
    }
    std::fclose(f);
    if (n < 2) {
        std::fprintf(stderr, "tsp: file %s has fewer than 2 cities\n", path);
        return -1;
    }
    return n;
}

/**
 * @brief Fills xy with n reproducible random cities in [0,1]^2 using a
 *        splitmix64 stream seeded from instance_seed. No global RNG state
 *        is touched, so calling this with the same seed on every MPI rank
 *        produces byte-identical city layouts.
 *
 * @param xy            Output buffer; caller must allocate 2*n*sizeof(float).
 * @param n             Number of cities to generate.
 * @param instance_seed Splitmix64 seed — must be identical on every rank
 *                      so all islands solve the same TSP instance.
 *
 * @returns void.
 *
 * @Structure
 *   - splitmix64 inner lambda; iterates 2*n times to fill x,y pairs
 */
static void tsp_random_instance(float* xy, int n, unsigned long long instance_seed) {
    unsigned long long s = instance_seed ? instance_seed : 0x9E3779B97F4A7C15ULL;
    auto next_unit = [&]() -> float {
        s += 0x9E3779B97F4A7C15ULL;
        unsigned long long z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z =  z ^ (z >> 31);
        return static_cast<float>(z >> 40) / static_cast<float>(1u << 24);
    };
    for (int i = 0; i < n; ++i) {
        xy[2 * i]     = next_unit();
        xy[2 * i + 1] = next_unit();
    }
}

/**
 * @brief Loads or generates a TSP city layout, uploads it to this GPU's
 *        constant memory via tsp_upload_instance, and configures cfg.
 *
 * @param tsp_file       Path to "x y" per-line instance file, or nullptr to
 *                       generate a random instance from instance_seed.
 * @param n_dims_hint    City count to generate when tsp_file is nullptr;
 *                       ignored when loading from file (n_dims set from file).
 * @param instance_seed  Seed passed to tsp_random_instance. Must be the same
 *                       on every MPI rank — do NOT add a rank offset here.
 * @param cfg            PSOConfig to update: n_dims <- city count,
 *                       bound_lo <- 0.0f, bound_hi <- 1.0f.
 *
 * @returns City count on success, or -1 on allocation / upload / range error.
 *
 * @Structure
 *   - malloc xy host buffer (2 * MAX_TSP_CITIES floats)
 *   - if tsp_file: tsp_load_file -> n
 *   - else:        validate n_dims_hint, tsp_random_instance -> n
 *   - tsp_upload_instance (cudaMemcpyToSymbol on this rank's GPU)
 *   - free xy, set cfg fields, return n
 */
static int setup_tsp_instance(const char* tsp_file,
                               int n_dims_hint,
                               unsigned long long instance_seed,
                               PSOConfig* cfg) {
    float* xy = static_cast<float*>(std::malloc(sizeof(float) * 2 * MAX_TSP_CITIES));
    if (!xy) { std::fprintf(stderr, "tsp: host alloc failed\n"); return -1; }

    int n;
    if (tsp_file) {
        n = tsp_load_file(tsp_file, xy, MAX_TSP_CITIES);
        if (n < 0) { std::free(xy); return -1; }
        std::printf("tsp: loaded %d cities from %s\n", n, tsp_file);
    } else {
        n = n_dims_hint;
        if (n < 2 || n > MAX_TSP_CITIES) {
            std::fprintf(stderr, "tsp: D=%d out of range (2..%d) for random instance\n",
                n, MAX_TSP_CITIES);
            std::free(xy);
            return -1;
        }
        tsp_random_instance(xy, n, instance_seed);
        std::printf("tsp: generated random instance of %d cities (seed=%llu)\n",
            n, (unsigned long long)instance_seed);
    }

    cudaError_t err = tsp_upload_instance(xy, n);
    std::free(xy);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "tsp: upload failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    //random-keys encoding: key order is all that matters, unit cube is natural
    cfg->n_dims   = n;
    cfg->bound_lo = 0.0f;
    cfg->bound_hi = 1.0f;
    return n;
}