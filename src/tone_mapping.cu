#include "tone_mapping.hpp"

__device__ glm::vec3 ToneMapper::aces_hill(glm::vec3 color) {
  glm::mat3 m1 = glm::mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383,
                           0.04823, 0.01566, 0.83777);
  glm::mat3 m2 = glm::mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813,
                           -0.07276, -0.07367, -0.00605, 1.07602);
  glm::vec3 v = m1 * color;
  glm::vec3 a = v * (v + 0.0245786f) - 0.000090537f;
  glm::vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
  return glm::pow(glm::clamp(m2 * (a / b), 0.0f, 1.0f), glm::vec3(1.0f / 2.2f));
}
