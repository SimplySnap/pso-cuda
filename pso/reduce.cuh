#pragma once

#include <stddef.h>
#include <cuda_runtime.h>

typedef struct {
    int   idx;
    float val;
} ReduceResult;

void reduce_argmin_cub(const float* pbest, int N,
                    void* tmp, size_t tmp_bytes,
                    ReduceResult* d_out, cudaStream_t s);
void reduce_argmin_custom(const float* pbest, int N,
                    void* tmp, size_t tmp_bytes,
                    ReduceResult* d_out, cudaStream_t s);
__global__ void kernel_copy_gbest_pos(
              const float* pbest_pos, float* gbest_pos,
            const ReduceResult* d_in, int N, int D);
// Same as kernel_copy_gbest_pos but reads the index from a bare int* — used
// for the once-per-run post-loop gather where the historical best lives in
// swarm.d_gbest_idx rather than the latest reduce output.
__global__ void kernel_gather_gbest_pos(
              const float* pbest_pos, float* gbest_pos,
            const int* d_idx, int N, int D);
size_t reduce_argmin_cub_workspace(int N);
