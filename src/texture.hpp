#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#else
typedef void float3;
#endif

class Texture {
public:
  Texture(size_t width, size_t height);
  ~Texture();
  GLuint get_id() const { return id; }
  size_t get_width() const { return width; }
  size_t get_height() const { return height; }
  struct cudaGraphicsResource *get_pbo_resource() const;
  void map_resource(float3 *&ptr);
  void unmap_resource();

private:
  GLuint id{0};
  GLuint pbo{0};
  struct cudaGraphicsResource *cuda_pbo_resource{nullptr};
  size_t width{32};
  size_t height{32};
};
