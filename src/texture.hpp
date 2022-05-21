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
typedef int cudaArray_t;
typedef int cudaSurfaceObject_t;
typedef int cudaTextureObject_t;
#endif

class Texture {
public:
  Texture(size_t width, size_t height);
  ~Texture();
  GLuint get_id() const { return id; }
  size_t get_width() const { return width; }
  size_t get_height() const { return height; }
  struct cudaGraphicsResource *get_img_resource() const;
  void map_resource(cudaArray_t &arr);
  void unmap_resource();
  cudaSurfaceObject_t create_cuda_surface_object();
  void destroy_cuda_surface_object(cudaSurfaceObject_t &obj);

  cudaTextureObject_t create_cuda_texture_object();
  void destroy_cuda_texture_object(cudaTextureObject_t &obj);

  void update();

private:
  GLuint id{0};
  struct cudaGraphicsResource *cuda_img_resource{nullptr};
  size_t width{32};
  size_t height{32};
};
