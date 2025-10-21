#!/bin/bash

# Double check if environment variables are already set
: "${__NV_PRIME_RENDER_OFFLOAD=1}"
: "${__GLX_VENDOR_LIBRARY_NAME=nvidia}"

# Force OpenGL to use NVIDIA GPU
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Build and run
exec cmake --build build | ./bin/mandelbrot
