#include "../include/Shaders.hpp"

#include <fstream>
#include <iterator>

std::string loadShader(const char* filePath) {
  std::ifstream in { filePath };
  std::string shaderString { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
  return shaderString;
}
