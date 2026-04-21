# pso-cuda
Parallel Implementation of Particle Swarm Optimization on CUDA

---

## Project Structure

```
pso-cuda/
├── CMakeLists.txt            ← Build system root
├── README.md
│
├── include/
│   └── pso/
│       ├── pso.h             ← PUBLIC API: PSO config, run function, types
│       └── evaluator.h       ← Function pointer / interface for objective functions
│
├── src/
│   ├── pso_kernel.cu         ← CUDA kernels (update velocity/position, global best reduction)
│   ├── pso_host.cu           ← Host orchestration: alloc, launch kernels, copy results
│   └── main.cu               ← Entry point: wires evaluator + calls pso_run()
│
├── evaluators/
│   ├── sphere.cu             ← Example: f(x) = sum(x_i^2)
│   ├── rosenbrock.cu         ← Example: banana function
│   └── evaluator_api.h       ← Shared typedef for device function pointer
│
└── tests/
    └── test_pso.cu           ← Correctness and convergence checks
```

---

## Parallelization Design

### Thread/Warp Mapping

This implementation uses a **warp-per-dimension, thread-per-particle** mapping (Idea 2):

- Each **thread block** covers a segment of the swarm — threads span particles within a dimension slice.
- Each **warp** is responsible for a single dimension index across all particles in the block.
- This satisfies **coalesced memory access** when using a Structure-of-Arrays (SoA) layout:
  `position[dim][particle]` — threads in the same warp read the same dimension across contiguous particles.

This is preferred over a warp-per-particle mapping for high-dimensional problems, where dimension-level parallelism is the bottleneck.

### Memory Layout (SoA)

All position, velocity, and personal best arrays are stored in **Structure-of-Arrays** order:

```
position[D][N]   — D dimensions, N particles
velocity[D][N]
pbest_pos[D][N]
pbest_fit[N]
```

This ensures that threads in a warp access contiguous memory addresses, maximizing memory throughput.

---

## Global Best (`gbest`) Strategy

Naively letting all threads write to a single `gbest` causes write collisions and cache contention. This implementation uses a **two-level hierarchical reduction**:

### Level 1 — Warp Shuffle Reduction (within a warp)

Each thread holds its particle's pre-computed fitness value `f_i` and its particle index `i`. A butterfly reduction using `__shfl_down_sync` finds the warp's local argmin — carrying both the fitness value and index in lockstep:

```cuda
float my_fit = fitness[threadIdx.x];
int   my_idx = threadIdx.x;

for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float other_f = __shfl_down_sync(0xFFFFFFFF, my_fit, offset);
    int   other_i = __shfl_down_sync(0xFFFFFFFF, my_idx, offset);
    if (other_f < my_fit) {
        my_fit = other_f;
        my_idx = other_i;
    }
}
// Lane 0 holds the warp-local best fitness and the index into the SoA position array
```

No re-evaluation of the objective function occurs here — only fitness comparisons on already-computed values.

### Level 2 — Block Reduction (shared memory) + Single Atomic Write

Warp winners (lane 0 of each warp) write their `(fitness, index)` pair to shared memory. A second pass reduces these into the single `block_best`. Then **one thread per block** attempts to update the true global `gbest` in global memory using `atomicCAS`:

```cuda
// Only one thread per block writes to global gbest
if (threadIdx.x == block_winner && block_fit < gbest_fit[0]) {
    atomicCAS(&gbest_lock, 0, 1);  // acquire
    if (block_fit < gbest_fit[0]) {
        gbest_fit[0] = block_fit;
        gbest_idx[0] = block_idx;
    }
    atomicExch(&gbest_lock, 0);    // release
}
```

This reduces global write contention from **N_particles writers** to **N_blocks writers** per iteration.

### Level 3 (optional) — Staging Array for Large Grids

When `N_blocks` is large, a follow-up reduction kernel reads one `block_best` per block from a staging array and reduces to the true `gbest` in a single thread — eliminating atomic contention entirely at the cost of one extra kernel launch per iteration.

---

## Asynchronous `gbest` Updates

Synchronizing `gbest` every iteration is correct but not always necessary. This implementation supports a **lazy global update** mode controlled by `pso_config.sync_interval`:

- Each block runs for `k` iterations using only its **block-local best**.
- Every `k` steps, block-local bests are flushed to the true global `gbest`.
- Larger `k` reduces synchronization overhead; smaller `k` preserves tighter convergence.

Recommended starting value: `k = 5–10`. Set `k = 1` to disable (synchronous mode).

---

## Multi-GPU / Island Model (Future Work)

The block-level hierarchy extends naturally to a multi-GPU design:

- Each GPU holds a **sub-swarm** in its own VRAM and maintains a local `gbest`.
- Every `T` iterations, sub-swarm bests are communicated across GPUs (via NVLink or host) and a global best is broadcast.
- Top-`m` particles can be migrated between sub-swarms to prevent premature convergence.

This mirrors the cooperative CPSO approach (van den Bergh & Engelbrecht, 2004) at the hardware level.

---

## Evaluator Interface

Objective functions are passed as **device function pointers** defined in `evaluators/evaluator_api.h`:

```c
typedef float (*DeviceEvaluator)(const float* position, int dims);
```

Built-in evaluators:
- `sphere.cu` — `f(x) = sum(x_i^2)`
- `rosenbrock.cu` — Banana function

Custom evaluators must be parallelized at the per-dimension level to avoid wasting GPU cycles on scalar CPU-bound computation. Each evaluator is launched as a sub-kernel or inlined within `pso_kernel.cu`.

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./pso_cuda
```

Requires CUDA 11.0+, CMake 3.18+, and a Volta or newer GPU (warp shuffle support).

---

## References

- Kennedy & Eberhart (1995). *Particle Swarm Optimization.* IEEE ICNN.
- van den Bergh & Engelbrecht (2004). *A Cooperative Approach to Particle Swarm Optimization.* IEEE TEC.
- NVIDIA (2022). *cuPSO: GPU Parallelization for PSO Algorithms.* ACM SAC.
- NVIDIA Developer Blog. *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics.*
