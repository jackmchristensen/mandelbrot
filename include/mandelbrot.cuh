#pragma once 

void gradient(char *pixels, int width, int height);

bool checkDevice();
void registerPixelBuffer(GLuint pbo);
void drawGradient(int width, int height);
