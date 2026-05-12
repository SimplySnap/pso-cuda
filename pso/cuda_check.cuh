#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
            __FILE__, __LINE__, cudaGetErrorString(_e));           \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)
