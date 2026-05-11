// --- REDUCTION (move to pso/reduce.cuh + pso/reduce.cu) ----------------------
//
// TODO(M3): struct ReduceResult { float val; int idx; };  // referenced in pso.h
//
// TODO(M3): reduce_argmin_cub(const float* pbest, int N,
//                             void* tmp, size_t tmp_bytes,
//                             ReduceResult* d_out, cudaStream_t s)
//           - Wraps cub::DeviceReduce::ArgMin.
//           - Use the two-call idiom in swarm_alloc to size tmp_bytes.
//
// TODO(M3): reduce_argmin_custom(...)
//           - L1: warp shuffle butterfly carrying (fit, idx) pair via
//                 __shfl_down_sync(0xFFFFFFFF, ...).
//           - L2: warp leaders write to shared mem; one warp reduces across warps.
//           - L3 (optional, large grids): staging array of block winners +
//                 second kernel for final argmin (avoid atomicCAS contention).
//
// TODO(M3): __global__ kernel_copy_gbest_pos(
//               const float* pbest_pos, float* gbest_pos,
//               const ReduceResult* d_in, int N, int D)
//           - D threads gather gbest_pos[d] = pbest_pos[d*N + d_in->idx].
//           - Only update if d_in->val < current gbest_val (host-side check OK).
//