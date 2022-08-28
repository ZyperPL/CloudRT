#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "texture.hpp"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <surface_indirect_functions.h>
#include <texture_indirect_functions.h>
#endif

struct RenderParameters
{
  int width { 128 };
  int height { 128 };
  glm::vec3 camera_position { 0.0f, 0.0f, 0.0f };
  glm::vec2 camera_rotation { 0.5f, 0.5f };
  glm::vec3 light_direction { 0.6f, 0.65f, -0.8f };
  glm::vec3 light_color { 1.0f, 0.9f, 0.6f };
  float light_power { 900.0f };
  float time { 1.0f };
  float density { 1.0f };
  glm::vec3 clouds_start { 300.0f, 2000.0f, 7000.0f };
  glm::vec3 clouds_end { 2000.0f, 8000.0f, 10000.0f };
  float gamma { 0.06f };
};

struct RenderMeasure
{
  bool enabled { false };
  float time_ms { 0.0f };
};

void launch_render(Texture &out_texture, Texture &clouds_texture, RenderParameters parameters, RenderMeasure &measure);
