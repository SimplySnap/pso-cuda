// Single-threaded CPU baseline for PSO benchmarking.
//
// This intentionally mirrors the current GPU loop order and SoA layout:
//   1. evaluate all particles and update pbest
//   2. reduce pbest to gbest
//   3. update every (dimension, particle) position/velocity entry
//
// RNG is std::mt19937, so this is a timing/sanity baseline rather than a
// bitwise correctness oracle for cuRAND-based GPU runs.

// USAGE: 
// (1) make cpu
// (2) ./build/pso_cpu [N] [D] [iters] [seed] [rastrigin|levy|schaffer_f2]
// Examples:
// (1) ./build/pso_cpu 1024 30 100 1234 rastrigin
// (2) ./build/pso_cpu 1024 30 100 1234 levy
// (3) ./build/pso_cpu 1024 2 100 1234 schaffer_f2

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
using Clock = std::chrono::steady_clock;

enum class EvaluatorKind {
    Rastrigin,
    Levy,
    SchafferF2,
};

struct Config {
    // Mirrors PSOConfig defaults used by pso_run/main.cu for the current GPU demo.
    int n_particles = 1024;
    int n_dims = 30;
    int max_iters = 100;
    unsigned int seed = 1234;
    EvaluatorKind evaluator = EvaluatorKind::Rastrigin;
    float w = 0.7f;
    float c1 = 1.5f;
    float c2 = 1.5f;
    float bound_lo = -5.12f;
    float bound_hi = 5.12f;
};

struct Timings {
    double eval_ms = 0.0;
    double reduce_ms = 0.0;
    double update_ms = 0.0;

    double total_ms() const {
        return eval_ms + reduce_ms + update_ms;
    }
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int parse_positive_int(const char* text, const char* name) {
    int value = 0;
    try {
        size_t parsed = 0;
        value = std::stoi(text, &parsed);
        if (parsed != std::string(text).size() || value <= 0) {
            throw std::invalid_argument("not positive");
        }
    } catch (const std::exception&) {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        std::exit(EXIT_FAILURE);
    }
    return value;
}

unsigned int parse_seed(const char* text) {
    try {
        size_t parsed = 0;
        unsigned long value = std::stoul(text, &parsed);
        if (parsed != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<unsigned int>(value);
    } catch (const std::exception&) {
        std::cerr << "Invalid seed: " << text << "\n";
        std::exit(EXIT_FAILURE);
    }
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

EvaluatorKind parse_evaluator(const char* text) {
    std::string value = lowercase(text);
    if (value == "rastrigin") return EvaluatorKind::Rastrigin;
    if (value == "levy") return EvaluatorKind::Levy;
    if (value == "schaffer" || value == "schaffer_f2" || value == "schaffer-f2" || value == "f2") {
        return EvaluatorKind::SchafferF2;
    }

    std::cerr << "Invalid evaluator: " << text
              << " (expected rastrigin, levy, or schaffer_f2)\n";
    std::exit(EXIT_FAILURE);
}

const char* evaluator_name(EvaluatorKind kind) {
    switch (kind) {
        case EvaluatorKind::Rastrigin: return "rastrigin";
        case EvaluatorKind::Levy: return "levy";
        case EvaluatorKind::SchafferF2: return "schaffer_f2";
    }
    return "unknown";
}

float square(float x) {
    return x * x;
}

float soa_at(const std::vector<float>& positions, int particle, int dim, int N) {
    return positions[dim * N + particle];
}

float rastrigin_cpu(const std::vector<float>& positions, int particle, int N, int D) {
    // Host mirror of evals/evals.cu::rastrigin_fn. The SoA read pattern matches
    // kernel_eval_and_pbest: positions[d * N + particle].
    float result = 10.0f * static_cast<float>(D);
    for (int d = 0; d < D; ++d) {
        float x = soa_at(positions, particle, d, N);
        result += x * x - 10.0f * std::cos(2.0f * kPi * x);
    }
    return result;
}

float levy_cpu(const std::vector<float>& positions, int particle, int N, int D) {
    // Host mirror of evals/evals.cu::levy_fn. The CUDA evaluator returns +inf
    // outside its supported dimensionality, so the CPU baseline does too.
    if (D < 3) {
        return std::numeric_limits<float>::infinity();
    }

    float x0 = soa_at(positions, particle, 0, N);
    float x_last = soa_at(positions, particle, D - 1, N);
    float y1 = 1.0f + (x0 - 1.0f) / 4.0f;
    float yn = 1.0f + (x_last - 1.0f) / 4.0f;

    float result = square(std::sin(kPi * y1));
    for (int d = 0; d < D - 1; ++d) {
        float x = soa_at(positions, particle, d, N);
        float x_next = soa_at(positions, particle, d + 1, N);
        float y = 1.0f + (x - 1.0f) / 4.0f;
        float yp = 1.0f + (x_next - 1.0f) / 4.0f;
        result += square(y - 1.0f) * (1.0f + 10.0f * square(std::sin(kPi * yp)));
    }

    result += square(yn - 1.0f) * (1.0f + square(std::sin(2.0f * kPi * yn)));
    return result;
}

float schaffer_f2_cpu(const std::vector<float>& positions, int particle, int N, int D) {
    // Host mirror of evals/evals.cu::schaffer_f2_fn, including the D == 2
    // requirement and the exact denominator used in the CUDA implementation.
    if (D != 2) {
        return std::numeric_limits<float>::infinity();
    }

    float x0 = soa_at(positions, particle, 0, N);
    float x1 = soa_at(positions, particle, 1, N);
    float numerator = square(std::sin(x0 * x0 - x1 * x1)) - 0.5f;
    float denominator = 1.0f + 0.001f * (x0 * x0 + x1 * x1);
    float frac = numerator / denominator;
    return 0.5f + frac * frac;
}

float evaluate_cpu(EvaluatorKind kind, const std::vector<float>& positions, int particle, int N, int D) {
    switch (kind) {
        case EvaluatorKind::Rastrigin:
            return rastrigin_cpu(positions, particle, N, D);
        case EvaluatorKind::Levy:
            return levy_cpu(positions, particle, N, D);
        case EvaluatorKind::SchafferF2:
            return schaffer_f2_cpu(positions, particle, N, D);
    }
    return std::numeric_limits<float>::infinity();
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (argc > 1) cfg.n_particles = parse_positive_int(argv[1], "N");
    if (argc > 2) cfg.n_dims = parse_positive_int(argv[2], "D");
    if (argc > 3) cfg.max_iters = parse_positive_int(argv[3], "iters");
    if (argc > 4) cfg.seed = parse_seed(argv[4]);
    if (argc > 5) cfg.evaluator = parse_evaluator(argv[5]);
    if (argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " [N] [D] [iters] [seed] [rastrigin|levy|schaffer_f2]\n";
        return EXIT_FAILURE;
    }

    const int N = cfg.n_particles;
    const int D = cfg.n_dims;
    const int n_entries = N * D;
    const float span = cfg.bound_hi - cfg.bound_lo;

    // Host mirrors of the swarm device buffers allocated in swarm_alloc.
    // The N * D arrays use the same SoA layout as the CUDA kernels.
    std::vector<float> positions(n_entries);
    std::vector<float> velocities(n_entries);
    std::vector<float> pbest_pos(n_entries);
    std::vector<float> fitness(N, std::numeric_limits<float>::infinity());
    std::vector<float> pbest(N, std::numeric_limits<float>::infinity());
    std::vector<float> gbest_pos(D, 0.0f);
    std::vector<float> gbest_history(cfg.max_iters, std::numeric_limits<float>::infinity());

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> init_pos(cfg.bound_lo, cfg.bound_hi);
    std::uniform_real_distribution<float> init_vel(-span, span);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    // Mirrors swarm_init/kernel_swarm_init: initialize positions, velocities,
    // and pbest_pos in SoA order with offset = dim * N + particle.
    for (int idx = 0; idx < n_entries; ++idx) {
        float pos = init_pos(rng);
        positions[idx] = pos;
        velocities[idx] = init_vel(rng);
        pbest_pos[idx] = pos;
    }

    float gbest_val = std::numeric_limits<float>::infinity();
    int gbest_idx = -1;
    Timings timings;

    for (int iter = 0; iter < cfg.max_iters; ++iter) {
        auto eval_start = Clock::now();
        // Mirrors kernel_eval_and_pbest: one logical worker per particle
        // calls the selected evaluator and updates pbest/pbest_pos when improved.
        for (int p = 0; p < N; ++p) {
            float fit = evaluate_cpu(cfg.evaluator, positions, p, N, D);
            fitness[p] = fit;
            if (fit < pbest[p]) {
                pbest[p] = fit;
                for (int d = 0; d < D; ++d) {
                    int offset = d * N + p;
                    pbest_pos[offset] = positions[offset];
                }
            }
        }
        timings.eval_ms += elapsed_ms(eval_start, Clock::now());

        auto reduce_start = Clock::now();
        // Mirrors reduce_argmin_cub over pbest plus kernel_commit_gbest.
        // On GPU, the reducer finds the best pbest index, then the commit
        // kernel copies that particle's pbest_pos into gbest_pos.
        for (int p = 0; p < N; ++p) {
            if (pbest[p] < gbest_val) {
                gbest_val = pbest[p];
                gbest_idx = p;
                for (int d = 0; d < D; ++d) {
                    gbest_pos[d] = pbest_pos[d * N + p];
                }
            }
        }

        // Mirrors kernel_commit_gbest writing s.d_gbest_history[iter].
        gbest_history[iter] = gbest_val;
        timings.reduce_ms += elapsed_ms(reduce_start, Clock::now());

        auto update_start = Clock::now();
        if (gbest_idx >= 0) {
            // Mirrors kernel_update: one logical worker per (dimension, particle)
            // updates velocity and position using the committed global best.
            for (int idx = 0; idx < n_entries; ++idx) {
                int d = idx / N;

                float r1 = unit(rng);
                float r2 = unit(rng);
                float pos = positions[idx];
                float vel = velocities[idx];
                float pb = pbest_pos[idx];
                float gb = gbest_pos[d];

                float new_vel = cfg.w * vel
                    + cfg.c1 * r1 * (pb - pos)
                    + cfg.c2 * r2 * (gb - pos);
                float new_pos = pos + new_vel;
                if (new_pos < cfg.bound_lo) {
                    new_pos = cfg.bound_lo;
                    new_vel = 0.0f;
                } else if (new_pos > cfg.bound_hi) {
                    new_pos = cfg.bound_hi;
                    new_vel = 0.0f;
                }

                positions[idx] = new_pos;
                velocities[idx] = new_vel;
            }
        }
        timings.update_ms += elapsed_ms(update_start, Clock::now());
    }

    // Mirrors the final PSOResult fields produced by pso_run after copying
    // d_gbest_val and gbest_pos back to host.
    std::cout << "impl,evaluator,N,D,iters,seed,eval_ms,reduce_ms,update_ms,total_ms,"
                 "final_gbest,achieved_bw_gbps,achieved_gflops\n";
    std::cout << "cpu,"
              << evaluator_name(cfg.evaluator) << ","
              << N << ","
              << D << ","
              << cfg.max_iters << ","
              << cfg.seed << ","
              << timings.eval_ms << ","
              << timings.reduce_ms << ","
              << timings.update_ms << ","
              << timings.total_ms() << ","
              << gbest_val << ","
              << "NA,NA\n";

    return EXIT_SUCCESS;
}
