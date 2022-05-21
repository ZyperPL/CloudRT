#pragma once

#include "glm/glm.hpp"
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
  int width;
  int height;
  glm::vec3 camera_position;
  glm::vec3 camera_direction;
  glm::vec3 light_position;
  glm::vec3 light_color;
  float time;
};

void launch_render(Texture &out_texture, Texture &clouds_texture, RenderParameters parameters);
