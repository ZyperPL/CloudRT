#include "test.hpp"

#include <cstdio>

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

void start_cuda_hello() {
  printf("Starting CUDA...\n");
  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("CUDA done.\n");
}

__device__ int get_value(int id)
{
#ifdef __CUDACC__
  //TODO: get sin
  return static_cast<int>((sin((double)(id) / 20000.0) + 1.0) * 120);
#else
  return id;
#endif
}

__global__ void render(uchar3 *d_out, int w, int h, int t) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h))
    return;                
  const int i = c + r * w; 

  //TODO: change d_out to double3
  int v = get_value(i + t * 256);
  d_out[i].x = (v + d_out[i].z) % 255;
  d_out[i].y = (get_value(i / t * 1024)) % 255;
  d_out[i].z = (d_out[i].y + i) % 255;
}

void launch_render(struct cudaGraphicsResource *pbo, size_t w, size_t h) {
  const dim3 blockSize(32, 32);
  const dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);
  uchar3 *d_out = 0;
  cudaGraphicsMapResources(1, &pbo, 0);
  cudaGraphicsResourceGetMappedPointer((void **)(&d_out), NULL, pbo);
  static int t = 0;
  t++;
  render<<<gridSize, blockSize>>>(d_out, w, h, t);
  cudaGraphicsUnmapResources(1, &pbo, 0);
}
