#include "render.hpp"

#include "noise_generator.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "tone_mapping.hpp"

__device__ const float EARTH_RADIUS = 6'371'230;

__device__ const int LIGHT_SAMPLES = 16;
__device__ const float MAX_LIGHT_DISTANCE = 500.0f;
__device__ const float LIGHT_STEP_LENGTH =
    MAX_LIGHT_DISTANCE / float(LIGHT_SAMPLES);

/*
 * sample_clouds
 * Samples clouds texture and returns cloud's density in the current layer
 */
__device__ float sample_clouds(glm::vec3 point, float cloud_height,
                               cudaTextureObject_t clouds_texture, int layer,
                               RenderParameters &parameters) {
  if (cloud_height < 0.1f || cloud_height > 0.9f)
    return 0.0f;
  point.z += parameters.time * 1.5f;

  float4 sample4 =
      tex2D<float4>(clouds_texture, point.z * -0.00005f, point.x * -0.00005f);
  float sample = layer == 0 ? sample4.x : layer == 1 ? sample4.y : sample4.z;

  float global_shape = glm::clamp((sample - 0.38f) * 5.0f, 0.0f, 2.0f);
  point.x += parameters.time * 0.5f;

  sample4 = tex2D<float4>(clouds_texture, 0.0002f * point.z, 0.0002f * point.x);
  sample = layer == 0 ? sample4.x : layer == 1 ? sample4.y : sample4.z;
  float shape = global_shape * glm::max(0.0f, sample - 0.28f) / 0.72f;
  shape *= glm::smoothstep(0.0f, 0.5f, cloud_height) *
           glm::smoothstep(1.0f, 0.5f, cloud_height);

  const float vertical_shape =
      pow(shape, 0.32f + 1.55f * glm::smoothstep(0.2f, 0.5f, cloud_height));

  if (vertical_shape <= 0.0f)
    return 0.0f;

  point.x += parameters.time * 2.0f;
  float density = glm::max(
      0.0f, vertical_shape - 0.7f * NoiseGenerator::fbm(point * 0.001f));

  if (density <= 0.0f)
    return 0.0f;

  // based on Andrew Schneider's work
  point.y += parameters.time * 3.0f;
  density -= 0.3f * NoiseGenerator::fbm(point * 0.005f) -
             0.1f * NoiseGenerator::fbm(point * 0.02f) -
             0.05f * NoiseGenerator::fbm(point * 0.06f);
  density = glm::max(0.0f, density);
  return global_shape * 0.24f * glm::min(1.0f, 5.0f * density);
}

namespace MiePhase {
// based on Thomas Schander's work
__device__ float numerical_mie_fit(float costh) {
  float bestParams[10];
  bestParams[0] = 9.805233e-06;
  bestParams[1] = -6.500000e+01;
  bestParams[2] = -5.500000e+01;
  bestParams[3] = 8.194068e-01;
  bestParams[4] = 1.388198e-01;
  bestParams[5] = -8.370334e+01;
  bestParams[6] = 7.810083e+00;
  bestParams[7] = 2.054747e-03;
  bestParams[8] = 2.600563e-02;
  bestParams[9] = -4.552125e-12;

  float p1 = costh + bestParams[3];
  glm::vec4 expValues = exp(
      glm::vec4(bestParams[1] * costh + bestParams[2], bestParams[5] * p1 * p1,
                bestParams[6] * costh, bestParams[9] * costh));
  glm::vec4 expValWeight =
      glm::vec4(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
  return glm::dot(expValues, expValWeight);
}

__device__ float henyey_greenstein(float mu, float inG) {
  return (1. - inG * inG) / (pow(1.0f + inG * inG - 2.0f * inG * mu, 1.5f) *
                             4.0f * glm::pi<double>());
}

__device__ float hillaire_mie(float c) {
  return glm::mix(henyey_greenstein(-0.5f, c), henyey_greenstein(0.8f, c),
                  0.5f);
}
}; // namespace MiePhase
__device__ float phase(float c) {
  return (MiePhase::numerical_mie_fit(c) + MiePhase::hillaire_mie(c)) / 6.0f;
}

/*
 * light_march
 * Performs ray-marching from the point inside the cloud to the light's source
 */
__device__ float light_march(glm::vec3 point, float phase_function,
                             float cloud_density, float light_d,
                             glm::vec3 sun_direction, float cloud_height,
                             cudaTextureObject_t clouds_texture, int layer,
                             RenderParameters &parameters) {

  float density = 0.0f;

  if (parameters.ray_noise_offset) {
    point +=
        sun_direction * LIGHT_STEP_LENGTH *
        NoiseGenerator::hash(glm::dot(point, glm::vec3(12.256f, 2.646f, 6.356f)) +
                             parameters.time_2);
  }
  for (int j = 0; j < LIGHT_SAMPLES; j++) {
    density +=
        sample_clouds(point + sun_direction * float(j) * LIGHT_STEP_LENGTH,
                      cloud_height, clouds_texture, layer, parameters);
  }

  const float scatter_amount =
      glm::mix(0.008f, 1.0f, glm::smoothstep(0.96f, 0.0f, light_d));
  const float lambert_beer_attenuation =
      exp(-LIGHT_STEP_LENGTH * density) +
      0.5f * scatter_amount * exp(-0.10f * LIGHT_STEP_LENGTH * density) +
      scatter_amount * 0.4f * exp(-0.02f * LIGHT_STEP_LENGTH * density);
  return lambert_beer_attenuation * phase_function *
         glm::mix(0.05f + 1.5f * glm::pow(glm::min(1.0f, cloud_density * 8.5f),
                                          0.3f + 5.5f * cloud_height),
                  1.0f, glm::clamp(density * 0.4f, 0.0f, 1.0f));
}

/*
 * sky_march
 * Performs ray-marching along the ray and checks cloud's density
 */
__device__ glm::vec3 sky_march(Ray &ray, cudaTextureObject_t clouds_texture,
                               RenderParameters &parameters) {
  glm::vec3 color = glm::vec3(0.0f);

  const Sphere atm_start_sphere{glm::vec3(0.0f, -EARTH_RADIUS, 0.0f),
                                EARTH_RADIUS + parameters.clouds_start.x};
  const Sphere atm_end_sphere{glm::vec3(0.0f, -EARTH_RADIUS, 0.0f),
                              EARTH_RADIUS + parameters.clouds_end.z};

  ray.pos = ray.origin;
  const float atmosphere_start_dst = atm_start_sphere.intersect(ray);
  const float atmosphere_end_dst = atm_end_sphere.intersect(ray);

  glm::vec3 point = ray.origin + atmosphere_start_dst * ray.dir;
  const float step_size = (atmosphere_end_dst - atmosphere_start_dst) /
                          float(parameters.ray_samples);

  if (parameters.ray_noise_offset) {
    point += ray.dir * step_size *
             NoiseGenerator::hash(
                 glm::dot(ray.dir, glm::vec3(12.256, 2.646, 6.356)) +
                 parameters.time_2);
  }

  float T = 1.0f;
  const float light_d = glm::dot(parameters.light_direction, ray.dir);
  const float phase_function = pow(phase(light_d), 4.0f) + phase(light_d);

  if (ray.dir.y > 0.0001f)
    for (int i = 0; i < parameters.ray_samples; i++) {
      int layer;
      float cloud_height;
      float point_h =
          length(point - glm::vec3(0.0f, -EARTH_RADIUS, 0.0f)) - EARTH_RADIUS;
      if (point_h < parameters.clouds_start.y) {
        cloud_height = glm::clamp(
            (point_h - parameters.clouds_start.x) /
                (parameters.clouds_end.x - parameters.clouds_start.x),
            0.0f, 1.0f);
        layer = 0;
      } else if (point_h < parameters.clouds_start.z) {
        cloud_height = glm::clamp(
            (point_h - parameters.clouds_start.y) /
                (parameters.clouds_end.y - parameters.clouds_start.y),
            0.0f, 1.0f);
        layer = 1;

      } else {
        cloud_height = glm::clamp(
            (point_h - parameters.clouds_start.z) /
                (parameters.clouds_end.z - parameters.clouds_start.z),
            0.0f, 1.0f);
        layer = 2;
      }
      const float density =
          sample_clouds(point, cloud_height, clouds_texture, layer, parameters);

      if (density > 0.0f) {
        float intensity = light_march(point, phase_function, density, light_d,
                                      parameters.light_direction, cloud_height,
                                      clouds_texture, layer, parameters);

        // based on Sebastien Hillarie's work
        const glm::vec3 ambient =
            (0.5f + 0.6f * cloud_height) * glm::vec3(0.2, 0.5, 1.0) * 6.5f +
            glm::vec3(0.8) * glm::max(0.0f, 1.0f - 2.0f * cloud_height);
        glm::vec3 radiance = ambient + parameters.light_color *
                                           parameters.light_power * intensity;
        radiance *= density * parameters.density;
        color += T * (radiance - radiance * glm::exp(-density * step_size)) /
                 density;
        T *= exp(-density * step_size);
        if (T <= 0.05f)
          break;
      }
      point += ray.dir * step_size;
    }

  Sphere upper_atmosphere{{0.0f, -EARTH_RADIUS, 0.0f},
                          EARTH_RADIUS + parameters.clouds_end.z + 1000.0f};
  color +=
      T * glm::vec3(3.0) *
      glm::max(0.0f, NoiseGenerator::fbm(
                         glm::vec3(1.0f, 1.0f, 1.8f) *
                         (point + upper_atmosphere.intersect(ray)) * 0.00005f) -
                         0.4f);

  glm::vec3 background =
      6.0f * glm::mix(glm::vec3(0.2f, 0.52f, 1.0f),
                      glm::vec3(0.8f, 0.95f, 1.0f),
                      glm::pow(0.5f + 0.5f * light_d, 15.0f)) +
      glm::mix(glm::vec3(3.5f), glm::vec3(0.0f),
               glm::min(1.0f, 2.3f * ray.dir.y));
  background += T * glm::vec3(1e4 * glm::smoothstep(0.9998f, 1.0f, light_d));
  color += background * T;

  return color;
}

__global__ void render_clouds(cudaSurfaceObject_t out_surface,
                              cudaTextureObject_t clouds_texture,
                              RenderParameters parameters) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= parameters.width) || (r >= parameters.height))
    return;
  [[maybe_unused]] const int i = c + r * parameters.width;

  double du = static_cast<double>(c) / static_cast<double>(parameters.width);
  double dv =
      1.0 - (static_cast<double>(r) / static_cast<double>(parameters.height));

  Ray ray;
  ray.origin = parameters.camera_position;
  ray.pos = ray.origin;

  const float camera_x_rotation = -7.0f * parameters.camera_rotation.x;
  glm::vec3 org =
      (glm::vec3(6.0f * glm::cos(camera_x_rotation),
                 glm::mix(1.2f, 10.0f, parameters.camera_rotation.y),
                 6.0f * glm::sin(camera_x_rotation)));
  glm::vec3 ta = glm::vec3(
      0.0f, glm::mix(4.2f, 15.0f, parameters.camera_rotation.y), 0.0f);

  glm::vec3 ww = glm::normalize(ta - org);
  glm::vec3 uu = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), ww));
  glm::vec3 vv = glm::normalize(glm::cross(ww, uu));
  glm::vec2 v = -1.0f + 2.0f * glm::vec2(du, dv);
  glm::vec3 dir = glm::normalize(v.x * uu + v.y * vv + 1.4f * ww);

  glm::vec3 color = glm::vec3(0.0f);

  parameters.light_direction = normalize(parameters.light_direction);
  org = ray.origin;
  ray.dir = dir;
  const float light_d = dot(parameters.light_direction, ray.dir);

  color = sky_march(ray, clouds_texture, parameters);
  ray.pos = ray.origin;
  Sphere fog_sphere{glm::vec3(0.0f, -EARTH_RADIUS, 0.0f),
                    EARTH_RADIUS + 160.0f};

  const float fog_phase = 0.5f * MiePhase::henyey_greenstein(light_d, 0.7f) +
                          0.5f * MiePhase::henyey_greenstein(light_d, -0.6f);

  glm::vec4 col = glm::vec4(
      glm::mix(fog_phase * 0.05f * glm::vec3(1.1f, 0.7f, 0.5f) *
                       parameters.light_color * parameters.light_power +
                   10.0f * glm::vec3(0.55f, 0.8f, 1.0f),
               color, glm::exp(-0.0003f * fog_sphere.intersect(ray))),
      1.0f);
  if (ray.dir.y < 0.0f)
    col *= 0.5f;

  if (parameters.aces) {
    col = glm::vec4(ToneMapper::aces_hill(glm::vec3(col.r, col.g, col.b) *
                                          parameters.gamma),
                    1.0f);
  } else {
    col = glm::vec4(glm::vec3(col.r, col.g, col.b) * parameters.gamma, 1.0f);
  }

  float4 output;
  output.x = col.r;
  output.y = col.g;
  output.z = col.b;
  output.w = 1.0f;

  if (parameters.image_blend_factor < 0.99f)
  {
    float4 previous;
    surf2Dread(&previous, out_surface, c * sizeof(float4), r);

    const float ibf = parameters.image_blend_factor;
    output.x = glm::clamp(output.x * ibf + previous.x * (1.0f - ibf), 0.0f, 1.0f);
    output.y = glm::clamp(output.y * ibf + previous.y * (1.0f - ibf), 0.0f, 1.0f);
    output.z = glm::clamp(output.z * ibf + previous.z * (1.0f - ibf), 0.0f, 1.0f);
  } else
  {
    output.x = glm::clamp(output.x, 0.0f, 1.0f);
    output.y = glm::clamp(output.y, 0.0f, 1.0f);
    output.z = glm::clamp(output.z, 0.0f, 1.0f);
  }

  surf2Dwrite(output, out_surface, c * sizeof(float4), r);
}

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void launch_render(Texture &out_texture, Texture &clouds_texture,
                   RenderParameters parameters, RenderMeasure &measure) {

  cudaEvent_t start, stop;
  if (measure.enabled) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  const dim3 blockSize(16, 16);
  const dim3 gridSize =
      dim3((parameters.width + blockSize.x - 1) / blockSize.x,
           (parameters.height + blockSize.y - 1) / blockSize.y);

  cudaSurfaceObject_t cuda_render_surface =
      out_texture.create_cuda_surface_object();
  cudaTextureObject_t cuda_clouds_texture =
      clouds_texture.create_cuda_texture_object();

  if (measure.enabled)
    cudaEventRecord(start);

  render_clouds<<<gridSize, blockSize>>>(cuda_render_surface,
                                         cuda_clouds_texture, parameters);

  if (measure.enabled)
    cudaEventRecord(stop);

  gpuErrchk(cudaPeekAtLastError());

  out_texture.destroy_cuda_surface_object(cuda_render_surface);
  clouds_texture.destroy_cuda_texture_object(cuda_clouds_texture);

  if (measure.enabled) {
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    measure.time_ms = ms;
  }
}
