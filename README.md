# pso-cuda
Parallel Implementation of Particle Swarm Optimization on CUDA

---

## Project Structure

## Project Structure
```
pso/
├── pso.h          — PSOConfig, swarm, PSOResult, pso_run()
├── pso.cu         — swarm lifecycle + main loop (calls kernels & reducer)
├── kernels.cuh    — __global__ kernel declarations (eval_pbest, update) + reduction
├── kernels.cu     — kernel implementations (eval, pbest, updating, reduction)
├── reduce.cuh     — ReduceResult, reduce_argmin_cub declaration
├── reduce.cu      — CUB-based argmin reduction + gbest commit/copy kernels
evals/
├── evals.cuh
└── evals.cu
src/
├── main.cu        — single-GPU entry point
├── main_ring.cu   — MPI ring-topology entry point
└── main_fc.cu     — MPI fully-connected entry point
mpi/
├── mpi_island.h   — IslandSyncData, sync callback declarations
└── mpi_island.cu  — island_gbest_exchange, island_migrate_ring/fc
```

---

## Parallelization Design

### Thread/Warp Mapping

This implementation uses a **flat thread-per-(particle × dimension) mapping**:

- Each **thread** is assigned a unique `(particle, dim)` pair via `idx = blockIdx.x * blockDim.x + threadIdx.x`, with `particle = idx % N` and `dim = idx / N`.
- Thread blocks cover a contiguous flat range of `N × D` entries — there is no explicit per-block dimension assignment.
- This satisfies **coalesced memory access** when using a Structure-of-Arrays (SoA) layout: `position[dim][particle]` — threads in the same warp share the same `dim` (consecutive `idx` values share `idx/N`) and read contiguous particle slots.

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

`gbest` is maintained as a **(fitness value, particle index)** pair stored in device memory. The update pipeline has two stages per iteration — no warp shuffles, shared-memory reductions, or atomic locks are involved.

### Stage 1 — CUB `DeviceReduce::ArgMin` over `pbest_fit[N]`

`reduce_argmin_cub` (in `reduce.cu`) wraps `cub::DeviceReduce::ArgMin` to find the particle index with the minimum personal-best fitness across all N particles:

```cuda
cub::DeviceReduce::ArgMin(
    tmp, tmp_bytes,
    pbest_fit,                                               // input: float[N]
    reinterpret_cast<cub::KeyValuePair<int,float>*>(d_out), // output: {idx, val}
    N, stream);
```

The temporary workspace is sized once at allocation time via the two-call CUB idiom (`reduce_argmin_cub_workspace`). `ReduceResult` is a `{int idx; float val;}` struct that is statically asserted to match `cub::KeyValuePair<int,float>` in memory layout.

### Stage 2 — `kernel_commit_gbest` (scalar, single thread)

A single-thread kernel reads the CUB output and conditionally updates the running global best


---

## Asynchronous `gbest` Updates

Synchronizing `gbest` every iteration is correct but not always necessary. This implementation supports a **lazy global update** mode controlled by `pso_config.sync_interval`:

- Each block runs for `k` iterations using only its **block-local best**.
- Every `k` steps, block-local bests are flushed to the true global `gbest`.
- Larger `k` reduces synchronization overhead; smaller `k` preserves tighter convergence.

Recommended starting value: `k = 5–10`. Set `k = 1` to disable (synchronous mode).

---

## Multi-GPU / Island Model

We also explore an island-model, where multiple GPUs run the algorithm, sharing their own swarm gbest periodically among the cluster. We use MPI to reduce and broadcast global bests.

- Each GPU holds a **sub-swarm** in its own VRAM and maintains a local `gbest`.
- Every `T` iterations, sub-swarm bests are communicated across GPUs (via MPI) and a global best is broadcast.

This mirrors the cooperative CPSO approach (van den Bergh & Engelbrecht, 2004) at the hardware level.

---

## Evaluator Interface

Objective functions are passed as **device function pointers** defined in `evaluators/evaluator_api.h`:

```c
typedef float (*DeviceEvaluator)(const float* position, int dims);
```

Custom evaluators must be parallelized at the per-dimension level to avoid wasting GPU cycles on scalar CPU-bound computation. Each evaluator is launched as a sub-kernel or inlined within `pso_kernel.cu`.

---

## Build

Requires CUDA 11.0+, an MPI installation (OpenMPI or MPICH), and a Volta or newer GPU.

```bash
# load toolchain on a SLURM cluster first:
# module load cuda gcc openmpi

# single-GPU build (no MPI required)
make
./build/pso_cuda --evaluator rastrigin --N 1024 --D 30 --iters 100

# MPI island builds
make mpi

# run ring topology (4 islands)
mpirun -np 4 ./build/pso_ring --evaluator rastrigin --N 1024 --D 30 --iters 100 --sync 10 --migrate 5

# run fully-connected topology (4 islands)
mpirun -np 4 ./build/pso_fc   --evaluator rastrigin --N 1024 --D 30 --iters 100 --sync 10 --migrate 5

# individual targets
make ring          # build pso_ring only
make fc            # build pso_fc only
make debug         # debug build (all targets)
make clean         # remove build/
make info          # print resolved paths and flags
```

---

## References

- Kennedy & Eberhart (1995). *Particle Swarm Optimization.* IEEE ICNN.
- van den Bergh & Engelbrecht (2004). *A Cooperative Approach to Particle Swarm Optimization.* IEEE TEC.
- NVIDIA (2022). *cuPSO: GPU Parallelization for PSO Algorithms.* ACM SAC.
- NVIDIA Developer Blog. *CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics.*
