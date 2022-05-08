#pragma once

#include "glm/glm.hpp"

struct Ray {
  glm::vec3 origin{0.0f, 0.0f, 0.0f};
  glm::vec3 pos{0.0f, 0.0f, 0.0f};
  glm::vec3 dir{0.0f, 0.0f, 0.0f};
};
