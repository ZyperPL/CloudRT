#include "texture.hpp"

const size_t COLOR_COMPONENTS_N = 3;

Texture::Texture(size_t width, size_t height) : width{width}, height{height} {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * COLOR_COMPONENTS_N * sizeof(GLfloat), 0, GL_STREAM_DRAW);

  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

Texture::~Texture()
{
  cudaGraphicsUnregisterResource(cuda_pbo_resource);
  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &id);
}

struct cudaGraphicsResource *Texture::get_pbo_resource() const 
{ 
  return cuda_pbo_resource; 
}

void Texture::map_resource(float3 *&ptr)
{
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)(&ptr), NULL, cuda_pbo_resource);
}

void Texture::unmap_resource()
{
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}
