#include "test.hpp"

#include <cstdio>

__global__ void cuda_hello()
{
  printf("Hello World from GPU!\n");
}

void start_cuda_hello() 
{
  printf("Starting CUDA...\n");
  cuda_hello<<<1,1>>>();
  cudaDeviceSynchronize();
  printf("CUDA done.\n");
}
