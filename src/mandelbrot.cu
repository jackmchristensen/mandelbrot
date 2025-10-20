#include <cstdio>
#include <cuda_runtime.h>

#include "../include/mandelbrot.cuh"

__global__ void gradient(uchar4 *pixels, int width, int height) {

}

void foo() {
  printf("Hello, World!\n"); 
}

// int main() {
//   int width = 1920;
//   int height = 1080;
//
//   uchar4 *pixels;
//
//   cudaMallocManaged(&pixels, width * height * sizeof(float));
//
//   dim3 blockSize = {16, 16};
//   dim3 numBlocks = {(uint)(width + 15)/16, (uint)(height + 15)/16};
//
//   gradient<<<blockSize, numBlocks>>>(pixels, width, height);
//
//   cudaDeviceSynchronize();
//
// }
