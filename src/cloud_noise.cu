#include "cloud_noise.hpp"

#include "glm/gtc/noise.hpp"

__device__ float perlin(const glm::vec3 &pos, float frequency,
                        int octaveCount) {
  const float octaveFrenquencyFactor = 2;

  float sum = 0.0f;
  float weightSum = 0.0f;
  float weight = 0.5f;
  for (int oct = 0; oct < octaveCount; oct++) {
    glm::vec4 p = glm::vec4(pos.x, pos.y, pos.z, 0.0f) * glm::vec4(frequency);
    float val = glm::perlin(p, glm::vec4(frequency));

    sum += val * weight;
    weightSum += weight;

    weight *= weight;
    frequency *= octaveFrenquencyFactor;
  }

  float noise = (sum / weightSum) * 0.5f + 0.5f;
  noise = std::fminf(noise, 1.0f);
  noise = std::fmaxf(noise, 0.0f);
  return noise;
}

__global__ void render(float3 *d_out, CloudsRenderParameters parameters) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= parameters.width) || (r >= parameters.height))
    return;

  const int i = c + r * parameters.width;
  double du = static_cast<double>(c) / static_cast<double>(parameters.width);
  double dv =
      1.0 - (static_cast<double>(r) / static_cast<double>(parameters.height));

  glm::vec4 col = glm::vec4(
      perlin(glm::vec3(parameters.position.x + du, parameters.position.y + dv,
                       parameters.position.z),
             parameters.frequency, parameters.octaves));

  float3 output;
  output.x = col.r;
  output.y = col.g;
  output.z = col.b;

  d_out[i].x = 0.0f;
  d_out[i].y = 0.0f;
  d_out[i].z = 0.0f;
  d_out[i] = output;
}

void generate_cloud_noise(Texture &texture, CloudsRenderParameters &params) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize =
      dim3((texture.get_width() + blockSize.x - 1) / blockSize.x,
           (texture.get_height() + blockSize.y - 1) / blockSize.y);

  float3 *d_out = nullptr;
  texture.map_resource(d_out);

  render<<<gridSize, blockSize>>>(d_out, params);

  texture.unmap_resource();
  cudaDeviceSynchronize();
}
