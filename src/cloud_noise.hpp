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
  float low_cut_l;
  float high_cut_l;
  float low_cut_m;
  float high_cut_m;
  float low_cut_h;
  float high_cut_h;
};


void generate_cloud_noise(Texture &texture, CloudsRenderParameters &parameters);
