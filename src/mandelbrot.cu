#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../include/mandelbrot.cuh"

namespace cuda {
  cudaGraphicsResource* cuda_pbo;
}

__global__ void gradient(uchar4* pixels, int width, int height) {

}

void registerPixelBuffer(GLuint pbo) {
  cudaGraphicsGLRegisterBuffer(&cuda::cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void drawGradient(int width, int height) {
  cudaGraphicsMapResources(1, &cuda::cuda_pbo);
  uchar4* d_pixels = nullptr;
  size_t bytes = 0;
  cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &bytes, cuda::cuda_pbo);

  dim3 block(16, 16);
  dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
  gradient<<<block, grid>>>(d_pixels, width, height);

  cudaGraphicsUnmapResources(1, &cuda::cuda_pbo);
}
