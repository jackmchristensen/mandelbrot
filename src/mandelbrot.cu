#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../include/mandelbrot.cuh"

namespace cuda {
  cudaGraphicsResource* cuda_pbo;
}

__global__ void gradient(uchar4* pixels, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  pixels[y*width + x] = make_uchar4((float(x)/width)*255, (float(y)/height)*255, 0, 255);
}

__device__ float hueToRGB(int p, int q, int t) {
  return 1.0f;
}

__global__ void mandelbrot(uchar4* pixels, int width, int height, float xmin, float xmax, float ymin, float ymax, int maxIter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  
  float u = (x + 0.5f) / float(width);
  float v = (y + 0.5f) / float(height);
  float cr = xmin + u * (xmax - xmin);
  float ci = ymin + v * (ymax - ymin);

  float zr = 0.0f, zi = 0.0f;
  int iter = 0;
  while (zr*zr + zi*zi < 4.0f && iter < maxIter) {
    float zr2 = zr*zr - zi*zi + cr;
    zi = 2.0f * zr * zi + ci;
    zr = zr2;
    iter++;
  }

  float t = iter / float(maxIter);

  unsigned char r = (unsigned char)(9.0f * (1 - t) * t*t*t * 255.0f);
  unsigned char g = (unsigned char)(15.0f * (1 - t) * (1 - t) * t*t * 255.0f);
  unsigned char b = (unsigned char)(8.5f * (1 - t) * (1 - t) * (1 - t) * t * 255.0f);

  if (iter == maxIter) { r = g = b = 0; }

  pixels[y*width+x] = make_uchar4(r, g, b, 255);
}

void registerPixelBuffer(GLuint pbo) {
  cudaGraphicsGLRegisterBuffer(&cuda::cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void drawGradient(int width, int height, float* center, float xRange, float yRange) {
  cudaGraphicsMapResources(1, &cuda::cuda_pbo);
  uchar4* d_pixels = nullptr;
  size_t bytes = 0;
  cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &bytes, cuda::cuda_pbo);

  float xmin = center[0] - (xRange * 0.5f);
  float xmax = center[0] + (xRange * 0.5f);
  float ymin = center[1] - (yRange * 0.5f);
  float ymax = center[1] + (yRange * 0.5f);

  dim3 block(16, 16);
  dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
  mandelbrot<<<grid, block>>>(d_pixels, width, height, xmin, xmax, ymin, ymax, 500);
  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
}
