#pragma once
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

class ToneMapper {
public:
  __device__ static glm::vec3 aces_hill(glm::vec3 color);
};
