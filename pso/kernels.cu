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