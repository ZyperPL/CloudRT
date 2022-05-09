#include "cloud_noise.hpp"

__device__ glm::vec3 perlin(glm::vec3 pos) {
  return pos;
  //TODO
}

__global__ void render(float3 *d_out, int width, int height) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= width) || (r >= height))
    return;

  const int i = c + r * width;
  double du = static_cast<double>(c) / static_cast<double>(width);
  double dv =
      1.0 - (static_cast<double>(r) / static_cast<double>(height));

  glm::vec4 col = glm::vec4(0.0f);
  col.r = (glm::sin(float(c)) + 1.0f) / 2.0f;
  col.g = glm::sin(du) * glm::cos(dv);
  col.b = (glm::sin(float(i)));

  float3 output;
  output.x = col.r;
  output.y = col.g;
  output.z = col.b;

  d_out[i].x = 0.0f;
  d_out[i].y = 0.0f;
  d_out[i].z = 0.0f;
  d_out[i] = output;
}

void generate_cloud_noise(Texture &texture)
{
  const dim3 blockSize(16, 16);
  const dim3 gridSize =
      dim3((texture.get_width() + blockSize.x - 1) / blockSize.x,
           (texture.get_height() + blockSize.y - 1) / blockSize.y);

  float3 *d_out = nullptr;
  texture.map_resource(d_out);

  render<<<gridSize, blockSize>>>(d_out, texture.get_width(), texture.get_height());

  texture.unmap_resource();
  cudaDeviceSynchronize();
}
