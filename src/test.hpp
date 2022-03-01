#pragma once

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#endif

void start_cuda_hello();
void launch_render(struct cudaGraphicsResource *pbo, size_t w, size_t h);
