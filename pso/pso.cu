
/*
Kernel structure:
┌─────────────────────────────────────────────────────┐
│  For each iteration:                                │
│                                                     │
│  1. EVALUATE       fitness[i] = f(positions[i])     │
│     1 thread per particle                           │
│                                                     │
│  2. UPDATE PBEST   if fitness[i] < pbest_fit[i]:    │
│                        pbest_pos[i] = positions[i]  │
│                        pbest_fit[i] = fitness[i]    │
│     Can fuse with step 1                            │
│                                                     │
│  3. REDUCE GBEST   gbest = min over all particles   │
│     Parallel reduction → single warp shuffle or    │
│     thrust::min_element                             │
│                                                     │
│  4. UPDATE V & X   per-element velocity + position  │
│     1 thread per (particle × dim)                  │
│     Needs RNG — cuRAND state per thread             │
└─────────────────────────────────────────────────────┘
*/

//meow


//block finds block-level best with cheap SMEM redux
//if cur better than gbest -> add fitness & positionto atomicAdd to shared queue
//then, read aux[] array, par tree redux over blocks to find true global best
//thread 0 block secretary
//

/*
pso.cu
│
├── swarm_alloc()       — cudaMalloc all arrays + reduce workspace
├── swarm_free()        — cudaFree everything
├── swarm_init()        — cuRAND init, random positions/velocities
│
├── pso_run()           ← main loop
│     │
│     ├── kernel_eval_and_pbest<<<>>>()    [kernels.cuh]
│     │
│     ├── reduce_argmin_cub()              [reduce.cuh / reduce.cu]
│     │     └── swap to reduce_argmin_custom() here with no other changes
│     │
│     ├── copy gbest_pos from positions[gbest_idx]
│     │
│     └── kernel_update<<<>>>()            [kernels.cuh]
│
└── pso_result_free()
*/
