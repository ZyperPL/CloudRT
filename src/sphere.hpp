#pragma once

#include "ray.hpp"

#include "glm/glm.hpp"

struct Sphere {
  glm::vec3 center;
  float radius;

  __device__ __host__ float intersect(const Ray &ray) const {

    glm::vec3 oc = ray.pos - center;
    const float b = 2.0f * glm::dot(ray.dir, oc);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float disc = b * b - 4.0f * c;
    if (disc < 0.0f)
      return -1.0f;
    const float q =
        (-b + ((b < 0.0f) ? -glm::sqrt(disc) : glm::sqrt(disc))) / 2.0f;
    float t0 = q;
    float t1 = c / q;
    if (t0 > t1) {
      const float temp = t0;
      t0 = t1;
      t1 = temp;
    }
    if (t1 < 0.0f)
      return -1.0f;

    return (t0 < 0.0f) ? t1 : t0;
  }
};
