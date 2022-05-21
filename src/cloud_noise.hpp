#pragma once

#include "glm/glm.hpp"
#include "texture.hpp"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <surface_indirect_functions.h>
#endif

struct CloudsRenderParameters
{
  glm::vec3 position;
  int width;
  int height;
  float frequency;
  float octaves;
  float time;
};


void generate_cloud_noise(Texture &texture, CloudsRenderParameters &parameters);
