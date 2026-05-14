// bench/pso_param_sweep.cpp
//
// Parameter sweep: c1/c2 ratio x N x {rastrigin, levy, schaffer_f2}
// Fixed: w=0.7, c1+c2=3.0, 4 runs per combo, N in {256,512,1024,2048}
// Ratio steps: c1/(c1+c2) in {0.1, 0.2, ..., 0.9}
//
// Output: pso_param_sweep.csv
//   function,N,c1,c2,ratio,run,iteration,gbest

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

// ── sweep knobs ──────────────────────────────────────────────────────────────
constexpr float W            = 0.7f;
constexpr float C_SUM        = 3.0f;   // c1 + c2 held constant
constexpr int   N_RUNS       = 4;
constexpr int   N_ITERS      = 500;

const std::vector<float> RATIOS = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f};
const std::vector<int>   N_VALS = {256, 512, 1024, 2048};

// Seeds for the 4 runs
constexpr unsigned int SEEDS[N_RUNS] = {1234u, 5678u, 9012u, 3456u};

// ── domains ──────────────────────────────────────────────────────────────────
struct FnSpec {
    std::string name;
    int   D;
    float lo, hi;
};

const std::vector<FnSpec> FUNCTIONS = {
    {"rastrigin",   30, -6.0f,   6.0f},
    {"levy",        30, -10.0f, 10.0f},
    {"schaffer_f2",  2, -100.0f, 100.0f},
};

// ── evaluators ───────────────────────────────────────────────────────────────
inline float sq(float x) { return x * x; }

inline float soa(const std::vector<float>& pos, int p, int d, int N) {
    return pos[d * N + p];
}

float eval_rastrigin(const std::vector<float>& pos, int p, int N, int D) {
    float r = 10.0f * static_cast<float>(D);
    for (int d = 0; d < D; ++d) {
        float x = soa(pos, p, d, N);
        r += x*x - 10.0f * std::cos(2.0f * kPi * x);
    }
    return r;
}

float eval_levy(const std::vector<float>& pos, int p, int N, int D) {
    if (D < 3) return std::numeric_limits<float>::infinity();
    float x0   = soa(pos, p, 0,   N);
    float xend = soa(pos, p, D-1, N);
    float y1   = 1.0f + (x0   - 1.0f) / 4.0f;
    float yn   = 1.0f + (xend - 1.0f) / 4.0f;
    float r = sq(std::sin(kPi * y1));
    for (int d = 0; d < D-1; ++d) {
        float y  = 1.0f + (soa(pos,p,d,  N) - 1.0f) / 4.0f;
        float yp = 1.0f + (soa(pos,p,d+1,N) - 1.0f) / 4.0f;
        r += sq(y - 1.0f) * (1.0f + 10.0f * sq(std::sin(kPi * yp)));
    }
    r += sq(yn - 1.0f) * (1.0f + sq(std::sin(2.0f * kPi * yn)));
    return r;
}

float eval_schaffer_f2(const std::vector<float>& pos, int p, int N) {
    float x0 = soa(pos, p, 0, N);
    float x1 = soa(pos, p, 1, N);
    float num  = sq(std::sin(x0*x0 - x1*x1)) - 0.5f;
    float den  = 1.0f + 0.001f * (x0*x0 + x1*x1);
    float frac = num / den;
    return 0.5f + frac * frac;
}

float evaluate(const FnSpec& fn,
               const std::vector<float>& pos, int p, int N) {
    if (fn.name == "rastrigin")   return eval_rastrigin(pos, p, N, fn.D);
    if (fn.name == "levy")        return eval_levy(pos, p, N, fn.D);
    if (fn.name == "schaffer_f2") return eval_schaffer_f2(pos, p, N);
    return std::numeric_limits<float>::infinity();
}

// ── single PSO run ────────────────────────────────────────────────────────────
std::vector<float> run_pso(const FnSpec& fn,
                            int N, float c1, float c2,
                            unsigned int seed) {
    const int D   = fn.D;
    const int NxD = N * D;
    const float span = fn.hi - fn.lo;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> rpos(fn.lo, fn.hi);
    std::uniform_real_distribution<float> rvel(-span, span);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    std::vector<float> pos(NxD), vel(NxD), pbest_pos(NxD);
    std::vector<float> pbest(N, std::numeric_limits<float>::infinity());
    std::vector<float> gbest_pos(D, 0.0f);
    std::vector<float> history(N_ITERS, std::numeric_limits<float>::infinity());

    for (int i = 0; i < NxD; ++i) {
        float p  = rpos(rng);
        pos[i]   = p;
        vel[i]   = rvel(rng);
        pbest_pos[i] = p;
    }

    float gbest_val = std::numeric_limits<float>::infinity();
    int   gbest_idx = -1;

    for (int iter = 0; iter < N_ITERS; ++iter) {
        // eval + pbest
        for (int p = 0; p < N; ++p) {
            float fit = evaluate(fn, pos, p, N);
            if (fit < pbest[p]) {
                pbest[p] = fit;
                for (int d = 0; d < D; ++d)
                    pbest_pos[d*N+p] = pos[d*N+p];
            }
        }
        // gbest reduce
        for (int p = 0; p < N; ++p) {
            if (pbest[p] < gbest_val) {
                gbest_val = pbest[p];
                gbest_idx = p;
                for (int d = 0; d < D; ++d)
                    gbest_pos[d] = pbest_pos[d*N+p];
            }
        }
        history[iter] = gbest_val;

        // update
        if (gbest_idx >= 0) {
            for (int idx = 0; idx < NxD; ++idx) {
                int   d  = idx / N;
                float r1 = unit(rng), r2 = unit(rng);
                float p_ = pos[idx], v_ = vel[idx];
                float pb = pbest_pos[idx], gb = gbest_pos[d];
                float nv = W*v_ + c1*r1*(pb-p_) + c2*r2*(gb-p_);
                float np = p_ + nv;
                if      (np < fn.lo) { np = fn.lo; nv = 0.0f; }
                else if (np > fn.hi) { np = fn.hi; nv = 0.0f; }
                pos[idx] = np;
                vel[idx] = nv;
            }
        }
    }
    return history;
}

} // namespace

int main() {
    std::ofstream csv("pso_param_sweep.csv");
    if (!csv) { std::cerr << "Cannot open pso_param_sweep.csv\n"; return 1; }
    csv << "function,N,c1,c2,ratio,run,iteration,gbest\n";

    const int total = static_cast<int>(
        FUNCTIONS.size() * N_VALS.size() * RATIOS.size() * N_RUNS);
    int done = 0;

    for (const auto& fn : FUNCTIONS) {
        for (int N : N_VALS) {
            for (float ratio : RATIOS) {
                float c1 = ratio * C_SUM;
                float c2 = (1.0f - ratio) * C_SUM;
                for (int r = 0; r < N_RUNS; ++r) {
                    auto hist = run_pso(fn, N, c1, c2, SEEDS[r]);
                    for (int iter = 0; iter < N_ITERS; ++iter) {
                        csv << fn.name << ","
                            << N       << ","
                            << c1      << ","
                            << c2      << ","
                            << ratio   << ","
                            << r       << ","
                            << iter    << ","
                            << hist[iter] << "\n";
                    }
                    ++done;
                    std::cerr << "[" << done << "/" << total << "]  "
                              << fn.name << "  N=" << N
                              << "  ratio=" << ratio
                              << "  run=" << r
                              << "  gbest=" << hist.back() << "\n";
                }
            }
        }
    }

    std::cout << "Written pso_param_sweep.csv\n";
    return 0;
}