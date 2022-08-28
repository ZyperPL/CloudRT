#include "cloud_noise.hpp"

#include "noise_generator.hpp"

#define surface_type float4

__device__ glm::vec4 generate_texture(double du, double dv, CloudsRenderParameters parameters)
{
  glm::vec4 col = glm::vec4(
      NoiseGenerator::perlin(glm::vec3(parameters.position.x + du, parameters.position.y + dv,
                       parameters.position.z),
             parameters.frequency, parameters.octaves));
  col *= glm::vec4(
      NoiseGenerator::perlin(glm::vec3(parameters.position.x * 1.0f + du, parameters.position.y * 1.0f + dv,
                       parameters.position.z * 1.0f),
             parameters.frequency * 0.5f, parameters.octaves));

  col += glm::vec4(
      NoiseGenerator::perlin(glm::vec3(parameters.position.x * 1.0f + du, parameters.position.y * 1.0f + dv,
                       parameters.position.z * 1.0f),
             parameters.frequency * 2.0f, parameters.octaves));
  col += glm::vec4(
      NoiseGenerator::worley(glm::vec3(parameters.position.x + du,
                            parameters.position.y + dv, parameters.position.z) * parameters.frequency,
                  parameters.frequency));

  col *= 0.33f;
  return col;
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

  glm::vec4 col = generate_texture(du, dv, parameters);
  parameters.position.x *= 0.41532f;
  parameters.position.y *= 0.6423f;
  parameters.position.z *= 0.154f;
  parameters.position.x += 5.41532f;
  parameters.position.y += 4.6423f;
  parameters.position.z += 1.154f;
  glm::vec4 col2 = generate_texture(du, dv, parameters);
  parameters.position.x *= 2.41532f;
  parameters.position.y *= 1.6423f;
  parameters.position.z *= 6.154f;
  parameters.position.x += 4.41532f;
  parameters.position.y += 1.6423f;
  parameters.position.z += 0.154f;
  parameters.position *= 4.0f;
  glm::vec4 col3 = generate_texture(du, dv, parameters);

  surface_type output;
  output.x = NoiseGenerator::remap(col.r, parameters.low_cut_l, parameters.high_cut_l, 0.0f, 1.0f);
  output.y = NoiseGenerator::remap(col2.g, parameters.low_cut_m, parameters.high_cut_m, 0.0f, 1.0f);
  output.z = NoiseGenerator::remap(col3.b, parameters.low_cut_h, parameters.high_cut_h, 0.0f, 1.0f);
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
