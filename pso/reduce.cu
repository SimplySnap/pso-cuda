// --- REDUCTION ----------------------
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

#include "pso.h"
#include "reduce.cuh"
#include "cuda_check.cuh"
#include <limits>

void reduce_argmin_cub(const float* pbest, int N,
                    void* tmp, size_t tmp_bytes,
                    ReduceResult* d_out, cudaStream_t s) {
  (void)pbest;
  (void)N;
  (void)tmp;
  (void)tmp_bytes;

  // Stub for teammate-owned CUB reduction. It keeps pso_run linkable but
  // deliberately reports "no valid best" until the real reducer is implemented.
  ReduceResult stub{};
  stub.val = std::numeric_limits<float>::infinity();
  stub.idx = -1;
  CUDA_CHECK(cudaMemcpyAsync(d_out, &stub, sizeof(ReduceResult),
      cudaMemcpyHostToDevice, s));
}

__global__ void kernel_copy_gbest_pos(
              const float* pbest_pos, float* gbest_pos,
            const ReduceResult* d_in, int N, int D) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;

  int best_idx = d_in->idx;
  if (best_idx < 0 || best_idx >= N) return;
  gbest_pos[d] = pbest_pos[d * N + best_idx];
}

size_t reduce_argmin_cub_workspace(int N) {
  (void)N;
  return sizeof(ReduceResult);
} 
