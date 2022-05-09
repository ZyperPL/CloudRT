#pragma once

#include "glm/glm.hpp"
#include "texture.hpp"

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#endif

void generate_cloud_noise(Texture &texture);
