#include "ray.hpp"

#include "render.hpp"

/*
Copyright (c) 2022 Kacper Zybała
Copyright (c) 2017 Thomas Schander

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/ 

struct TextureView
{
  float3 *data;
  int width;
  int height;
};

// based on https://www.shadertoy.com/view/4dSBDt
__device__ const float PI = 3.141592;

__device__ const float EARTH_RADIUS = 6300e3;
__device__ const float CLOUD_START = 600.0;
__device__ const float CLOUD_HEIGHT = 500.0;
__device__ const glm::vec3 SUN_POWER = glm::vec3(1.0, 0.9, 0.6) * 900.0f;
__device__ const glm::vec3 LOW_SCATTER = glm::vec3(1.1, 0.7, 0.5);

__device__ float hash(float n) { return glm::fract(sin(n) * 43758.5453); }

// float hash(vec2 p) {
//     return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123);
// }

__device__ float noise(glm::vec3 x) {
  glm::vec3 p = floor(x);
  glm::vec3 f = fract(x);
  f = f * f * (3.0f - 2.0f * f);
  // return glm::sin(x.x) + cos(x.y) + glm::sin(x.z)*cos(x.z) / 453.234;
  return p.x*0.00001f + f.z*0.00001f + 0.1f;
  // return textureLod(iChannel2, (p+f+0.5)/24.0, 0.0).x;
}

__device__ float fbm(glm::vec3 p) {
  glm::mat3 m =
      glm::mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
  float f;
  f = 0.5000f * noise(p);
  p = m * p * 2.02f;
  f += 0.2500f * noise(p);
  p = m * p * 2.03f;
  f += 0.1250f * noise(p);
  p = m * p * 2.04f;
  return f;
}

__device__ float intersectSphere(glm::vec3 origin, glm::vec3 dir,
                                 glm::vec3 spherePos, float sphereRad) {
  glm::vec3 oc = origin - spherePos;
  float b = 2.0 * glm::dot(dir, oc);
  float c = glm::dot(oc, oc) - sphereRad * sphereRad;
  float disc = b * b - 4.0 * c;
  if (disc < 0.0)
    return -1.0;
  float q = (-b + ((b < 0.0) ? -sqrt(disc) : sqrt(disc))) / 2.0;
  float t0 = q;
  float t1 = c / q;
  if (t0 > t1) {
    float temp = t0;
    t0 = t1;
    t1 = temp;
  }
  if (t1 < 0.0)
    return -1.0;

  return (t0 < 0.0) ? t1 : t0;
}

__device__ float3 sample(TextureView &tex, glm::vec3 pos)
{
  float fx = glm::abs(pos.x * tex.width);
  float fy = glm::abs(pos.y * tex.height);
  int x = int(fx) % tex.width;
  int y = int(fy) % tex.height;
  int idx00 = glm::clamp(x + y * tex.width, 0, tex.width * tex.height);
  int idx01 = glm::clamp(x + ((y+1) % tex.height) * tex.width, 0, tex.width * tex.height);
  int idx10 = glm::clamp((x+1) % tex.width + y * tex.width, 0, tex.width * tex.height);
  int idx11 = glm::clamp((x+1) % tex.width + (y+1) % tex.height * tex.width, 0, tex.width * tex.height);
  float v00 = tex.data[idx00].x;
  float v01 = tex.data[idx01].x;
  float v10 = tex.data[idx10].x;
  float v11 = tex.data[idx11].x;
  printf("%f %f %d %d, %d %d %d %d, %f %f %f %f\n", fx, fy, x, y, idx00, idx01, idx10, idx11, v00, v01, v10, v11);
  return tex.data[idx00];
  float vy = v00 * (fy - float(y)) + v01 * (1.0f - fy - float(y));
  float vx = v11 * (fx - float(x)) + v10 * (1.0 - fx - float(x));
}

__device__ float clouds(glm::vec3 p, float &cloudHeight, cudaTextureObject_t clouds_texture) {
  float atmoHeight =
      length(p - glm::vec3(0.0f, -EARTH_RADIUS, 0.0f)) - EARTH_RADIUS;
  cloudHeight =
      glm::clamp((atmoHeight - CLOUD_START) / (CLOUD_HEIGHT), 0.0f, 1.0f);
  // p.z += iTime*10.3;
  //int idx = int(0.02f * (p.z * clouds_texture.width + p.x));
  //idx = int(0.00005f * p.x * 1024.0f + 0.00005f * p.z) % (1024*1024);
  //printf("p=%f, %f, %f, idx=%d\n", p.x, p.y, p.z, idx);
  
  float largeWeather =
      glm::clamp((glm::sin(-0.005f * glm::vec2(p.z, p.x)).x +
                  glm::sin(-0.005f * glm::vec2(p.y, p.x)).y - 0.18f) *
                     5.0f,
                 0.0f, 2.0f);
  // p.x += iTime*8.3;
//  largeWeather = sample(clouds_texture, p).x;

  //return sample(clouds_texture, p * 0.0010f).x;
  float weather = largeWeather * (glm::max(0.0f, 1.0f)-0.28f) / 0.72f;
  weather *= glm::smoothstep(0.0f, 0.5f, cloudHeight) *
             glm::smoothstep(1.0f, 0.5f, cloudHeight);
  float cloudShape =
      pow(weather, 0.3f + 1.5f * glm::smoothstep(0.2f, 0.5f, cloudHeight));
  if (cloudShape <= 0.0)
    return 0.0;
  // p.x += iTime*12.3;
  float den = glm::max(0.0f, cloudShape - 0.7f * fbm(p * .01f));
  if (den <= 0.0)
    return 0.0;

  // p.y += iTime*15.2;
  den = glm::max(0.0f, den - 0.3f * fbm(p * 0.05f) + 0.1f * fbm(p * 0.093f) +
                           0.05f * fbm(p * 0.041f));
  return largeWeather * 0.2f * glm::min(1.0f, 5.0f * den);
}

// From https://www.shadertoy.com/view/4sjBDG
__device__ float numericalMieFit(float costh) {
  float bestParams[10];
  bestParams[0] = 9.805233e-06;
  bestParams[1] = -6.500000e+01;
  bestParams[2] = -5.500000e+01;
  bestParams[3] = 8.194068e-01;
  bestParams[4] = 1.388198e-01;
  bestParams[5] = -8.370334e+01;
  bestParams[6] = 7.810083e+00;
  bestParams[7] = 2.054747e-03;
  bestParams[8] = 2.600563e-02;
  bestParams[9] = -4.552125e-12;

  float p1 = costh + bestParams[3];
  glm::vec4 expValues = exp(
      glm::vec4(bestParams[1] * costh + bestParams[2], bestParams[5] * p1 * p1,
                bestParams[6] * costh, bestParams[9] * costh));
  glm::vec4 expValWeight =
      glm::vec4(bestParams[0], bestParams[4], bestParams[7], bestParams[8]);
  return glm::dot(expValues, expValWeight);
}

__device__ float HenyeyGreenstein(float mu, float inG) {
  return (1. - inG * inG) /
         (pow(1. + inG * inG - 2.0 * inG * mu, 1.5) * 4.0 * PI);
}

__device__ float Hillaire(float c) {
  return glm::mix(HenyeyGreenstein(-0.5f, c), HenyeyGreenstein(0.8f, c), 0.5f);
}

__device__ float phase(float c) {
  return (numericalMieFit(c) + Hillaire(c) + HenyeyGreenstein(c, 0.8f) * 2.0f) /
         4.0f;
}

__device__ float lightRay(glm::vec3 p, float phaseFunction, float dC, float mu,
                          glm::vec3 sun_direction, float cloudHeight, cudaTextureObject_t clouds_texture) {
  int nbSampleLight = 32;
  float zMaxl = 700.;
  float stepL = zMaxl / float(nbSampleLight);

  float lighRayDen = 0.0;
  // p += sun_direction*stepL*hash(glm::dot(p, glm::vec3(12.256, 2.646, 6.356))
  // + iTime);
  for (int j = 0; j < nbSampleLight; j++) {
    float cloudHeight;
    lighRayDen +=
        clouds(p + sun_direction * float(j) * stepL / 1.0f, cloudHeight, clouds_texture);
  }

  float scatterAmount =
      glm::mix(0.008f, 1.0f, glm::smoothstep(0.96f, 0.0f, mu));
  float beersLaw = exp(-stepL * lighRayDen) +
                   0.5 * scatterAmount * exp(-0.1 * stepL * lighRayDen) +
                   scatterAmount * 0.4 * exp(-0.02 * stepL * lighRayDen);
  return beersLaw * phaseFunction *
         glm::mix(0.05f + 1.5f * glm::pow(glm::min(1.0f, dC * 8.5f),
                                          0.3f + 5.5f * cloudHeight),
                  1.0f, glm::clamp(lighRayDen * 0.4f, 0.0f, 1.0f));
}

__device__ glm::vec3 skyRay(glm::vec3 org, glm::vec3 dir,
                            glm::vec3 sun_direction, cudaTextureObject_t clouds_texture) {
  const float ATM_START = EARTH_RADIUS + CLOUD_START;
  const float ATM_END = ATM_START + CLOUD_HEIGHT;

  int nbSample = 1024;
  glm::vec3 color = glm::vec3(0.0);
  float distToAtmStart =
      intersectSphere(org, dir, glm::vec3(0.0, -EARTH_RADIUS, 0.0), ATM_START);
  float distToAtmEnd =
      intersectSphere(org, dir, glm::vec3(0.0, -EARTH_RADIUS, 0.0), ATM_END);
  glm::vec3 p = org + distToAtmStart * dir;
  float stepS = (distToAtmEnd - distToAtmStart) / float(nbSample);
  float T = 1.;
  float mu = glm::dot(sun_direction, dir);
  float phaseFunction = pow(phase(mu), 4.0) + phase(mu);

  // p += dir*stepS*hash(glm::dot(dir, glm::vec3(12.256, 2.646, 6.356)) +
  // iTime);
  if (dir.y > 0.00)
    for (int i = 0; i < nbSample; i++) {
      float cloudHeight;
      float density = clouds(p, cloudHeight, clouds_texture);

      if (density > 0.0) {
        float intensity =
            lightRay(p, phaseFunction, density, mu, sun_direction, cloudHeight, clouds_texture);
        glm::vec3 ambient =
            (0.5f + 0.6f * cloudHeight) * glm::vec3(0.2, 0.5, 1.0) * 6.5f +
            glm::vec3(0.8) * glm::max(0.0f, 1.0f - 2.0f * cloudHeight);
        glm::vec3 radiance = ambient + SUN_POWER * intensity;
        radiance *= density;
        color += T * (radiance - radiance * glm::exp(-density * stepS)) /
                 density; // By Seb Hillaire
        T *= exp(-density * stepS);
        if (T <= 0.05f)
          break;
      }
      p += dir * stepS;
    }

  glm::vec3 pC =
      org + intersectSphere(org, dir, glm::vec3(0.0f, -EARTH_RADIUS, 0.0f),
                            ATM_END + 1000.0f) *
                dir;
  color +=
      T * glm::vec3(3.0) *
      glm::max(0.0f, fbm(glm::vec3(1.0f, 1.0f, 1.8f) * pC * 0.002f) - 0.4f);
  glm::vec3 background =
      6.0f * glm::mix(glm::vec3(0.2f, 0.52f, 1.0f),
                      glm::vec3(0.8f, 0.95f, 1.0f),
                      glm::pow(0.5f + 0.5f * mu, 15.0f)) +
      glm::mix(glm::vec3(3.5f), glm::vec3(0.0f), glm::min(1.0f, 2.3f * dir.y));
  background += T * glm::vec3(1e4 * glm::smoothstep(0.9998f, 1.0f, mu));
  color += background * T;

  return color;
}

__device__ glm::vec3 aces_tonemap(glm::vec3 color) {
  glm::mat3 m1 = glm::mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383,
                           0.04823, 0.01566, 0.83777);
  glm::mat3 m2 = glm::mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813,
                           -0.07276, -0.07367, -0.00605, 1.07602);
  glm::vec3 v = m1 * color;
  glm::vec3 a = v * (v + 0.0245786f) - 0.000090537f;
  glm::vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
  return glm::pow(glm::clamp(m2 * (a / b), 0.0f, 1.0f), glm::vec3(1.0f / 2.2f));
}

__global__ void render(cudaSurfaceObject_t out_surface, cudaTextureObject_t clouds_texture, RenderParameters parameters) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= parameters.width) || (r >= parameters.height))
    return;
  const int i = c + r * parameters.width;

  double du = static_cast<double>(c) / static_cast<double>(parameters.width);
  double dv =
      1.0 - (static_cast<double>(r) / static_cast<double>(parameters.height));

  Ray ray;
  ray.origin = parameters.camera_position;
  ray.pos = ray.origin;
  ray.dir = glm::normalize(glm::vec3(du * 2.0f - 1.0f, dv * 2.0f - 1.0f, 1.0));

  glm::vec2 q = glm::vec2(du, dv);
  glm::vec2 v = -1.0f + 2.0f * q;
  // v.x *= iResolution.x/ iResolution.y;
  // glm::vec2 mo = iMouse.xy / iResolution.xy;
  glm::vec2 mo = glm::vec2(0.5f, 0.5f);
  float camRot = -7.0 * mo.x;
  glm::vec3 org =
      (glm::vec3(6.0f * glm::cos(camRot), glm::mix(1.2f, 10.0f, mo.y),
                 6.0f * glm::sin(camRot)));
  glm::vec3 ta = glm::vec3(0.0f, glm::mix(4.2f, 15.0f, mo.y), 0.0f);

  glm::vec3 ww = glm::normalize(ta - org);
  glm::vec3 uu = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), ww));
  glm::vec3 vv = glm::normalize(glm::cross(ww, uu));
  glm::vec3 dir = glm::normalize(v.x * uu + v.y * vv + 1.4f * ww);
  glm::vec3 color = glm::vec3(.0);
  glm::vec3 sun_direction = normalize(glm::vec3(0.6, 0.65, -0.8));
  float fogDistance = intersectSphere(
      org, dir, glm::vec3(0.0f, -EARTH_RADIUS, 0.0f), float(EARTH_RADIUS));
  float mu = dot(sun_direction, dir);

  // Sky
  if (fogDistance == -1.) {
    color = skyRay(org, dir, sun_direction, clouds_texture);
    fogDistance = intersectSphere(org, dir, glm::vec3(0.0, -EARTH_RADIUS, 0.0),
                                  EARTH_RADIUS + 160.0);
  }

  float fogPhase =
      0.5f * HenyeyGreenstein(mu, 0.7) + 0.5 * HenyeyGreenstein(mu, -0.6);
  glm::vec4 col = glm::vec4(glm::mix(fogPhase * 0.1f * LOW_SCATTER * SUN_POWER +
                                         10.0f * glm::vec3(0.55, 0.8, 1.0),
                                     color, glm::exp(-0.0003f * fogDistance)),
                            1.0f);
  col = glm::vec4(aces_tonemap(glm::vec3(col.r, col.g, col.b) * 0.06f), 1.0f);

  float4 output;
  output.x = col.r;
  output.y = col.g;
  output.z = col.b;
  output.w = 1.0f;

  {
    float4 cl;
    cl = tex2D<float4>(clouds_texture, du, dv);
    output.x = cl.x * 2.0f;
  }

  surf2Dwrite(output, out_surface, c * sizeof(float4), r);
}

void launch_render(Texture &out_texture, 
                   Texture &clouds_texture,
                   RenderParameters parameters) {
  const dim3 blockSize(16, 16);
  const dim3 gridSize =
      dim3((parameters.width + blockSize.x - 1) / blockSize.x,
           (parameters.height + blockSize.y - 1) / blockSize.y);

  cudaSurfaceObject_t cuda_render_surface = out_texture.create_cuda_surface_object();
  cudaTextureObject_t cuda_clouds_texture = clouds_texture.create_cuda_texture_object();

  render<<<gridSize, blockSize>>>(cuda_render_surface, cuda_clouds_texture, parameters);

  out_texture.destroy_cuda_surface_object(cuda_render_surface);
  clouds_texture.destroy_cuda_texture_object(cuda_clouds_texture);
}
