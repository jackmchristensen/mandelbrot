#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
#endif

#include <GL/gl.h>      
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>

#include "../include/mandelbrot.cuh"

namespace cuda {
  cudaGraphicsResource* cuda_pbo;
}

// Takes width and height in pixels and calculates a portion of the Mandelbrot set defined by xmin, xmax, ymin, and ymax
// Draws to pixels
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

  // Color Mandelbrot set
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

// If the pixel buffer changes you must unregister it before reregistering it with updated size
void UnregisterPixelBuffer() {
  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
  cuda::cuda_pbo = nullptr;
}

// Wrapper function for CUDA mandelbrot() function
// Width and height are the dimensions of the texture in pixels
// Center is the center position in Mandelbrot coordinates
// xRange and yRange represent the dimensions of the Mandelbrot set
void drawMandelbrot(int width, int height, double* center, double xRange, double yRange, int maxIterations) {
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // Tell CUDA to write output to the pixel buffer
  cudaGraphicsMapResources(1, &cuda::cuda_pbo);
  uchar4* d_pixels = nullptr;
  size_t bytes = 0;
  cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &bytes, cuda::cuda_pbo);

  // Calculate min and max values for Mandelbrot set
  double xmin = center[0] - (xRange * 0.5f);
  double xmax = center[0] + (xRange * 0.5f);
  double ymin = center[1] - (yRange * 0.5f);
  double ymax = center[1] + (yRange * 0.5f);

  // 256 blocks
  // 1 thread per pixel
  dim3 block(16, 16);
  dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

  // cudaEventRecord(start, 0);
  mandelbrot<<<grid, block>>>(d_pixels, width, height, xmin, xmax, ymin, ymax, maxIterations);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);

  // float ms = 0.0f;
  // cudaEventElapsedTime(&ms, start, stop);
  // printf("Kernel time: %.3fms\n", ms);

  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
}
