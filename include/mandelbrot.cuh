#pragma once 

void gradient(char *pixels, int width, int height);

bool checkDevice();
void registerPixelBuffer(GLuint pbo);
void UnregisterPixelBuffer();
void drawMandelbrot(int width, int height, double* center, double xRange, double yRange, int maxIterations);
