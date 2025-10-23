#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>   // must come before <GL/gl.h> on Windows
#endif

#include <GL/gl.h>       // gives you GLuint, GLenum, etc.
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>

#include "../include/mandelbrot.cuh"

namespace cuda {
  cudaGraphicsResource* cuda_pbo;
}

__global__ void mandelbrot(uchar4* pixels, int width, int height, double xmin, double xmax, double ymin, double ymax, int maxIter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  
  double u = (x + 0.5) / double(width);
  double v = (y + 0.5) / double(height);
  double cr = xmin + u * (xmax - xmin);
  double ci = ymin + v * (ymax - ymin);

  double zr = 0.0f, zi = 0.0;
  int iter = 0;
  while (zr*zr + zi*zi < 4.0 && iter < maxIter) {
    double zr2 = zr*zr - zi*zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr2;
    iter++;
  }

  double t = iter / double(maxIter);

  unsigned char r = (unsigned char)(9.0f * (1 - t) * t*t*t * 255.0f);
  unsigned char g = (unsigned char)(15.0f * (1 - t) * (1 - t) * t*t * 255.0f);
  unsigned char b = (unsigned char)(8.5f * (1 - t) * (1 - t) * (1 - t) * t * 255.0f);

  if (iter == maxIter) { r = g = b = 0; }

  pixels[y*width+x] = make_uchar4(r, g, b, 255);
}

void registerPixelBuffer(GLuint pbo) {
  cudaGraphicsGLRegisterBuffer(&cuda::cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void UnregisterPixelBuffer() {
  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
  cuda::cuda_pbo = nullptr;
}

void drawMandelbrot(int width, int height, double* center, double xRange, double yRange) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaGraphicsMapResources(1, &cuda::cuda_pbo);
  uchar4* d_pixels = nullptr;
  size_t bytes = 0;
  cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &bytes, cuda::cuda_pbo);

  double xmin = center[0] - (xRange * 0.5f);
  double xmax = center[0] + (xRange * 0.5f);
  double ymin = center[1] - (yRange * 0.5f);
  double ymax = center[1] + (yRange * 0.5f);

  dim3 block(16, 16);
  dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

  cudaEventRecord(start, 0);
  mandelbrot<<<grid, block>>>(d_pixels, width, height, xmin, xmax, ymin, ymax, 200);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  // printf("Kernel time: %.3fms\n", ms);

  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
}
