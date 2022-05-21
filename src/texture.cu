#include "texture.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

Texture::Texture(size_t width, size_t height) : width{width}, height{height} {
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
               GL_FLOAT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  cudaGraphicsGLRegisterImage(&cuda_img_resource, id, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

Texture::~Texture() {
  cudaGraphicsUnregisterResource(cuda_img_resource);
  glDeleteTextures(1, &id);
}

void Texture::update() {
  glBindTexture(GL_TEXTURE_2D, id);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT,
                  NULL);
}

struct cudaGraphicsResource *Texture::get_img_resource() const {
  return cuda_img_resource;
}

void Texture::map_resource(cudaArray_t &arr) {
  cudaGraphicsMapResources(1, &cuda_img_resource, 0);
  cudaGraphicsSubResourceGetMappedArray(&arr, cuda_img_resource, 0, 0);
}

void Texture::unmap_resource() {
  cudaGraphicsUnmapResources(1, &cuda_img_resource, 0);
}

cudaSurfaceObject_t Texture::create_cuda_surface_object() {
  cudaArray_t arr;
  map_resource(arr);

  struct cudaResourceDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr;

  cudaSurfaceObject_t obj = 0;
  cudaCreateSurfaceObject(&obj, &desc);
  return obj;
}

void Texture::destroy_cuda_surface_object(cudaSurfaceObject_t &obj) {
  unmap_resource();
  cudaDestroySurfaceObject(obj);
}

cudaTextureObject_t Texture::create_cuda_texture_object() {
  cudaArray_t arr;
  map_resource(arr);

  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = arr;

  cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(cudaTextureDesc));

  tex_desc.normalizedCoords = 1;
  tex_desc.filterMode = cudaFilterModeLinear;

  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.addressMode[2] = cudaAddressModeClamp;

  tex_desc.readMode = cudaReadModeElementType;

  cudaTextureObject_t obj = 0;
  cudaCreateTextureObject(&obj, &res_desc, &tex_desc, NULL);
  return obj;
}

void Texture::destroy_cuda_texture_object(cudaTextureObject_t &obj) {
  unmap_resource();
  cudaDestroyTextureObject(obj);
}
