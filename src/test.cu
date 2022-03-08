#include "test.hpp"

#include <cstdio>

#include "glm/glm.hpp"

#include "cuda_noise.cuh"

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

void start_cuda_hello() {
  printf("Starting CUDA...\n");
  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("CUDA done.\n");
}

struct Ray
{
  glm::vec3 pos{0.0f, 0.0f, 0.0f};
  glm::vec3 dir{0.0f, 0.0f, 0.0f};
};

__device__ __host__ bool intersect(Ray ray, glm::vec3 ball, float radius)
{
  size_t STEPS = 1000;
  while (STEPS --> 0)
  {
    ray.pos += ray.dir * 0.5f;
    if (glm::distance(ray.pos, ball) < radius)
      return true;
  }

  return false;
}

__global__ void render(float3 *d_out, int w, int h, float t) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h))
    return;
  const int i = c + r * w;

  Ray ray;
  ray.pos.x = (double)c / (double)w;
  ray.pos.y = (double)r / double(h);
  ray.pos.z = 0.0f;

  glm::vec3 light{0.0f, -15.0f, 70.0f};

  glm::vec3 camera{0.0f, 0.0f, 0.0f};
  camera.x = 0.5f;
  camera.y = 0.5f;
  camera.z = -1.0f;

  ray.dir = glm::normalize(ray.pos - camera);

  d_out[i].x = 0.0f;
  d_out[i].y = 0.0f;
  d_out[i].z = 0.0f;

  size_t STEPS = 1000;
  while (STEPS --> 0)
  {
    ray.pos += ray.dir * 0.1f;
    glm::vec3 ball = glm::vec3(0.0f, -10.0f - sin(t / 4.0f) * 2.0f, 80.0f);
    if (glm::distance(ray.pos, ball) < 10.0f)
    {
      ray.dir = glm::reflect(ray.dir, glm::normalize(ray.pos - ball));
      ray.dir = glm::normalize(ray.dir);
      d_out[i].x += glm::clamp(glm::dot(ray.dir, glm::normalize(ray.pos - light)), 0.0f, 1.0f) * (1.0f / glm::distance(ray.pos, light));
      ray.pos += ray.dir;
    }
    if (ray.pos.y > 2.0f)
    {
      ray.dir = glm::reflect(ray.dir, glm::vec3(0.0f, 1.0f, 0.0f));

      Ray light_ray;
      light_ray.pos = ray.pos;
      light_ray.dir = glm::normalize(light - ray.pos);

      float light_intensity = glm::clamp(glm::dot(glm::vec3(0.0f, 1.0f, 0.0f), -light_ray.dir), 0.0f, 1.0f);

      if (intersect(light_ray, ball, 8.0f))
        light_intensity = 0.1f;
        
      d_out[i].z = (sin(ray.pos.x) * 2.0f + cos(ray.pos.z) * 2.0f) * light_intensity;

      ray.pos += ray.dir;
    }

    float noise = cudaNoise::perlinNoise(float3{ray.pos.x, ray.pos.y, ray.pos.z + t * 100.0f}, 0.1f, 1234);
    if (noise > 0.8f)
    {
      d_out[i].x += noise / (float)(STEPS);
    }
  }

  float3 pos;
  pos.x = float(c) / float(w);
  pos.y = float(r) / float(h);
  pos.z = t / 100.0f;
  float n = (cudaNoise::perlinNoise(pos, 1.0, 123) + 1.0f) / 3.0f;
  n += (cudaNoise::perlinNoise(pos, 10.0, 123) + 1.0f) / 3.0f;
  n += (cudaNoise::perlinNoise(pos, 30.0, 123) + 1.0f) / 3.0f;
  //d_out[i].x = n / 3.0f;
}

void launch_render(struct cudaGraphicsResource *pbo, size_t w, size_t h) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize = dim3((w + blockSize.x - 1) / blockSize.x,
                             (h + blockSize.y - 1) / blockSize.y);
  float3 *d_out = 0;
  cudaGraphicsMapResources(1, &pbo, 0);
  cudaGraphicsResourceGetMappedPointer((void **)(&d_out), NULL, pbo);
  static float t = 0.0f;
  t += 0.1f;
  render<<<gridSize, blockSize>>>(d_out, w, h, t);
  cudaGraphicsUnmapResources(1, &pbo, 0);
}
