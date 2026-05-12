#pragma once

#include <stddef.h>
#include <cuda_runtime.h>

typedef struct {
    float val;
    int   idx;
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
size_t reduce_argmin_cub_workspace(int N);
