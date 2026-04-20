# pso-cuda
Parallel Implementation of Particle Swarm Optimization



Main implementation structure:
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