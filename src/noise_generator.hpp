#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/noise.hpp"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <surface_indirect_functions.h>
#include <texture_indirect_functions.h>
#endif

class NoiseGenerator {
public:
  __device__ static float hash(float n) {
    return glm::fract(sin(n) * 43758.5453);
  }

  __device__ static float hash(glm::vec3 p) {
    p = glm::fract(p * 0.3183099f + 0.1f);
    p *= 17.0f;
    return glm::fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  __device__ static float noise(const glm::vec3 &x) {
    glm::vec3 i = glm::floor(x);
    glm::vec3 f = glm::fract(x);
    f = f * f * (3.0f - 2.0f * f);

    return glm::mix(glm::mix(glm::mix(hash(i + glm::vec3(0.0f, 0.0f, 0.0f)),
                                      hash(i + glm::vec3(1, 0, 0)), f.x),
                             glm::mix(hash(i + glm::vec3(0, 1, 0)),
                                      hash(i + glm::vec3(1, 1, 0)), f.x),
                             f.y),
                    glm::mix(glm::mix(hash(i + glm::vec3(0, 0, 1)),
                                      hash(i + glm::vec3(1, 0, 1)), f.x),
                             glm::mix(hash(i + glm::vec3(0, 1, 1)),
                                      hash(i + glm::vec3(1, 1, 1)), f.x),
                             f.y),
                    f.z);
  }

  __device__ static float fbm(glm::vec3 p) {
    glm::mat3 m =
        glm::mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
    float f;
    f = 0.5000f * noise(p);
    p = m * p * 2.02f;
    f += 0.2500f * noise(p);
    p = m * p * 2.03f;
    f += 0.1250f * noise(p);
    p = m * p * 2.04f;
    return f;
  }

  __device__ static float remap(float domain, float min_x, float max_x,
                                float min_y, float max_y) {
    return (((domain - min_x) / (max_x - min_x)) * (max_y - min_y)) + min_y;
  }

  __device__ static glm::vec3 hash33(glm::vec3 p3) {
    p3 = fract(p3 * glm::vec3(0.1031f, 0.11369f, 0.13787f));
    p3 += dot(p3, glm::vec3(p3.y, p3.x, p3.z) + 19.19f);
    return -1.0f + 2.0f * glm::fract(glm::vec3((p3.x + p3.y) * p3.z,
                                               (p3.x + p3.z) * p3.y,
                                               (p3.y + p3.z) * p3.x));
  }

  __device__ static float worley(glm::vec3 uv, float freq) {
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

    return 1. - minDist;
  }

  __device__ static float perlin(const glm::vec3 &pos, float frequency,
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
};
