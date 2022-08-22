#include "cloud_noise.hpp"

#include "glm/gtc/noise.hpp"

#define surface_type float4

__device__ float remap(float domain, float min_x, float max_x, float min_y, float max_y) {
  return (((domain - min_x) / (max_x - min_x)) * (max_y - min_y)) + min_y;
}

__device__ glm::vec3 hash33(glm::vec3 p3) {
  p3 = fract(p3 * glm::vec3(0.1031f, 0.11369f, 0.13787f));
  p3 += dot(p3, glm::vec3(p3.y, p3.x, p3.z) + 19.19f);
  return -1.0f +
         2.0f * glm::fract(glm::vec3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y,
                                     (p3.y + p3.z) * p3.x));
}

__device__ float worleyNoise(glm::vec3 uv, float freq) {
  glm::vec3 id = glm::floor(uv);
  glm::vec3 p = glm::fract(uv);

  float minDist = 10000.;
  for (float x = -1.; x <= 1.; ++x) {
    for (float y = -1.; y <= 1.; ++y) {
      for (float z = -1.; z <= 1.; ++z) {
        glm::vec3 offset = glm::vec3(x, y, z);
        glm::vec3 h =
            hash33(glm::mod(id + offset, glm::vec3(freq))) * 0.5f + 0.5f;
        h += offset;
        glm::vec3 d = p - h;
        minDist = glm::min(minDist, glm::dot(d, d));
      }
    }
  }

  // inverted worley noise
  return 1. - minDist;
}

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

__global__ void render(cudaSurfaceObject_t surface,
                       CloudsRenderParameters parameters) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= parameters.width) || (r >= parameters.height))
    return;

  [[maybe_unused]] const int i = c + r * parameters.width;
  double du = static_cast<double>(c) / static_cast<double>(parameters.width);
  double dv =
      1.0 - (static_cast<double>(r) / static_cast<double>(parameters.height));

  glm::vec4 col = glm::vec4(
      perlin(glm::vec3(parameters.position.x + du, parameters.position.y + dv,
                       parameters.position.z),
             parameters.frequency, parameters.octaves));

  col *= glm::vec4(
      perlin(glm::vec3(parameters.position.x * 0.03151f + du, parameters.position.y * 0.0454f + dv,
                       parameters.position.z),
             parameters.frequency, parameters.octaves));

  col += glm::vec4(
      perlin(glm::vec3(parameters.position.x * 0.6591f + du, parameters.position.y * 0.4564f + dv,
                       parameters.position.z),
             parameters.frequency, parameters.octaves));

  glm::vec4 col2 = glm::vec4(perlin(
      glm::vec3(parameters.position.x * 3.12f + du,
                parameters.position.y * 341.f + dv, parameters.position.z),
      parameters.frequency, parameters.octaves));

  glm::vec4 col3 = glm::vec4(perlin(
      glm::vec3(parameters.position.x * 0.12f + du,
                parameters.position.y * 0.2f + dv, parameters.position.z),
      parameters.frequency, parameters.octaves));

  col += glm::vec4(
      worleyNoise(glm::vec3(parameters.position.x + du,
                            parameters.position.y + dv, parameters.position.z) * parameters.frequency,
                  parameters.frequency));

  col *= 0.25;

  col2 = col;
  col3 = col;

  surface_type output;
  output.x = remap(col.r, parameters.low_cut_l, parameters.high_cut_l, 0.0f, 1.0f);
  output.y = remap(col2.g, parameters.low_cut_m, parameters.high_cut_m, 0.0f, 1.0f);
  output.z = remap(col3.b, parameters.low_cut_h, parameters.high_cut_h, 0.0f, 1.0f);
  // output.y = col.g;
  // output.z = col.b;
  output.w = 1.0f;

  surf2Dwrite(output, surface, c * sizeof(surface_type), r);
}

void generate_cloud_noise(Texture &texture, CloudsRenderParameters &params) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize =
      dim3((texture.get_width() + blockSize.x - 1) / blockSize.x,
           (texture.get_height() + blockSize.y - 1) / blockSize.y);

  cudaSurfaceObject_t surface_obj = texture.create_cuda_surface_object();

  render<<<gridSize, blockSize>>>(surface_obj, params);

  texture.destroy_cuda_surface_object(surface_obj);

  cudaDeviceSynchronize();
}
