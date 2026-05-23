// --- REDUCTION ----------------------
// TODO(M3): reduce_argmin_cub(const float* pbest, int N,
//                             void* tmp, size_t tmp_bytes,
//                             ReduceResult* d_out, cudaStream_t s)
//           - Wraps cub::DeviceReduce::ArgMin.
//           - Use the two-call idiom in swarm_alloc to size tmp_bytes.
//
// __global__ kernel_copy_gbest_pos(
//               const float* pbest_pos, float* gbest_pos,
//               const ReduceResult* d_in, int N, int D)
//           - D threads gather gbest_pos[d] = pbest_pos[d*N + d_in->idx].
//           - Only update if d_in->val < current gbest_val (host-side check OK).
//

#include "pso.h"
#include "reduce.cuh"
#include "cuda_check.cuh"
#include <limits>
#include <cub/cub.cuh>

void reduce_argmin_cub(const float* pbest, int N,
                    void* tmp, size_t tmp_bytes,
                    ReduceResult* d_out, cudaStream_t s) {
  /*
  Uses cub library to find the ArgMin of pbest - passed in to comparison function to update
  gbest.
  Runs reduction on each iteration. Takes fitness[N] arr, calls argmin with pre-alloc tmp buff.
  Then, passes result to KeyValuePair<int,float> (index, fitness value) and converts to
  'ReduceResult' data type.
  stream input s - lets us overlap reduction with other independent work
  */
  // ReduceResult must be { int idx; float val; } to match KeyValuePair<int,float>
    static_assert(sizeof(ReduceResult) == sizeof(cub::KeyValuePair<int,float>),
                  "ReduceResult layout mismatch with KeyValuePair<int,float>");                    
  CUDA_CHECK(cub::DeviceReduce::ArgMin(
    tmp, tmp_bytes,
    pbest,
    reinterpret_cast<cub::KeyValuePair<int,float>*>(d_out),
    N, s));
}

__global__ void kernel_copy_gbest_pos(
              const float* pbest_pos, float* gbest_pos,
            const ReduceResult* d_in, int N, int D) {
  /*
  Copy pbest position of 'winning' particle into gbest.
  Check if 'winning' handled externally. d_in gives us location of winning particle
  */
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;

  int best_idx = d_in->idx;
  if (best_idx < 0 || best_idx >= N) return;
  gbest_pos[d] = pbest_pos[d * N + best_idx];
}

__global__ void kernel_gather_gbest_pos(
              const float* pbest_pos, float* gbest_pos,
            const int* d_idx, int N, int D) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  int idx = *d_idx;
  if (idx < 0 || idx >= N) return;
  gbest_pos[d] = pbest_pos[d * N + idx];
}

size_t reduce_argmin_cub_workspace(int N) {
  /*
  Helper function gives us number of bytes needed for argmin_cub temp buffer
  Makes use of cub ArgMin properties: when nullptr passed in, returns number of bytes needed
  */
  size_t bytes = 0;
  cub::DeviceReduce::ArgMin(nullptr, bytes,
      (float*)nullptr,
      (cub::KeyValuePair<int,float>*)nullptr, N);
  return bytes;
}