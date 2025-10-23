#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <iostream>
#include <chrono>

#include "../include/mandelbrot.cuh"
#include "../include/UpdateFlags.hpp"
#include "../include/Shaders.hpp"

namespace win {
  int width = 1280;
  int height = 720;
};

namespace scene {
  namespace gl {
    extern const int kVersionMajor = 3;
    extern const int kVersionMinor = 3;
  };
 
  struct Image {
    double xRange;
    double yRange;
    double center[2];
  };

  float scrollZoomMult = 0.1f;
  float keyZoomMult = 2.0f;
};

double LinearInterpolation(double a, double b, float amt) {
  return a*double(amt) + b*(1.0-double(amt));
}

void ConvertToMandelbrotCoord(double* newX, double* newY, double curX, double curY, double centerX, double centerY, double rangeX, double rangeY) {
  *newX = curX * rangeX - (rangeX * 0.5) + centerX;
  *newY = curY * rangeY - (rangeY * 0.5) - centerY;
}

void ZoomToMouse(scene::Image* image, SDL_Window* window, float zoomMult) {
  float xMouse, yMouse;
  SDL_GetMouseState(&xMouse, &yMouse);

  int width, height;
  SDL_GetWindowSizeInPixels(window, &width, &height);

  double xMouse_0_1 = zoomMult >= 1.0 ? double(xMouse) / width : 1.0 - (double(xMouse) / width);
  double yMouse_0_1 = zoomMult >= 1.0 ? double(yMouse) / height : 1.0 - (double(yMouse) / height);
  xMouse_0_1 = xMouse_0_1 * image->xRange - (image->xRange * 0.5) + image->center[0];
  yMouse_0_1 = yMouse_0_1 * image->yRange - (image->yRange * 0.5) - image->center[1];

  image->center[0] = zoomMult >= 1.0 ? LinearInterpolation(image->center[0], xMouse_0_1, 1.0f / zoomMult) : LinearInterpolation(image->center[0], xMouse_0_1, zoomMult);
  image->center[1] = zoomMult >= 1.0 ? LinearInterpolation(image->center[1], -yMouse_0_1, 1.0f / zoomMult) : LinearInterpolation(image->center[1], -yMouse_0_1, zoomMult);

  image->xRange *= 1.0 / zoomMult;
  image->yRange *= 1.0 / zoomMult;
}

void RenderMandelbrot(scene::Image image, GLuint textureBuffer, GLuint pixelBuffer) { 
  drawMandelbrot(win::width, win::height, image.center, image.xRange, image.yRange);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
  glBindTexture(GL_TEXTURE_2D, textureBuffer); 
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win::width, win::height, GL_RGBA, GL_UNSIGNED_BYTE, reinterpret_cast<const void*>(0));
}

int main() {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::cerr << "Failed to initialize SDL" << std::endl;
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, scene::gl::kVersionMajor);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, scene::gl::kVersionMinor);
  
  SDL_Window* window = SDL_CreateWindow("Mandelbrot", win::width, win::height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
  if (!window) {
    std::cerr << "Failed to create window" << std::endl;
    return -1;
  }
  auto glContext = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, glContext);
  SDL_GL_SetSwapInterval(0);
  
  glewExperimental = GL_TRUE;
  if(glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  scene::Image image = {
    2.0 * (float(win::width) / win::height),                              // xRange
    2.0,                                                                  // yRange
    { (-(image.xRange*(2.0/3.0))+(image.xRange*(1.0/3.0))) / 2.0, 0.0 }   // center
  };

  auto vendor   = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
  auto renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  std::cout << "GL_VENDOR=" << (vendor?vendor:"?") 
          << "  GL_RENDERER=" << (renderer?renderer:"?") << "\n";

  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  std::string vertexString = loadShader("shaders/vertex.glsl");
  const char* vertexSource = vertexString.c_str();
  glShaderSource(vertexShader, 1, &vertexSource, nullptr); 
  glCompileShader(vertexShader);

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  std::string fragmentString = loadShader("shaders/fragment.glsl");
  const char* fragmentSource = fragmentString.c_str();
  glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
  glCompileShader(fragmentShader);

  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
  GLint ok = GL_FALSE; glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if(!ok) {
    std::cerr << "Failed to link program" << std::endl;
    return -1;
  }

  glDetachShader(program, vertexShader);
  glDeleteShader(vertexShader);
  glDetachShader(program, fragmentShader);
  glDeleteShader(fragmentShader);

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, win::width, win::height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, win::width * win::height * 4, nullptr, GL_DYNAMIC_DRAW);

  registerPixelBuffer(pbo);

  glUseProgram(program);
  GLint loc = glGetUniformLocation(program, "u_tex");
  if (loc == -1) std::cerr << "u_tex not found" << std::endl;
  glUniform1i(loc, 0);

  float timer = 0.0f;
  SDL_GL_SwapWindow(window);
  UpdateFlags flags = None;
  bool isRunning = true;
  while (isRunning){
    auto start = std::chrono::high_resolution_clock::now();

    glViewport(0, 0, win::width, win::height);
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_EVENT_KEY_DOWN:
          switch (event.key.scancode) {
            case SDL_SCANCODE_ESCAPE:
              isRunning = false;
              break;
            case SDL_SCANCODE_EQUALS:
              // TODO move ZoomToMouse out of switch statement
              ZoomToMouse(&image, window, scene::keyZoomMult);
              flags |= Render;
              continue;
            case SDL_SCANCODE_MINUS:
              ZoomToMouse(&image, window, 1.0f / scene::keyZoomMult);
              flags |= Render;
              continue;
            default:
              continue;
          }
        case SDL_EVENT_MOUSE_WHEEL:
          if (event.wheel.y > 0.01f) {
            ZoomToMouse(&image, window, 1.0f + (event.wheel.y * scene::scrollZoomMult));
          } else if (event.wheel.y < -0.01f){
            ZoomToMouse(&image, window, 1.0f + (event.wheel.y * scene::scrollZoomMult));
          }
          flags |= Render;

          continue;
        case SDL_EVENT_WINDOW_RESIZED:
          flags |= Render | Resize;
          continue;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
          flags |= Render | Resize;
          continue;
        case SDL_EVENT_QUIT:
          isRunning = false;
          break;
      } 
    }

    if ((flags & Resize) == Resize) {
      SDL_GetWindowSizeInPixels(window, &win::width, &win::height);
      image.xRange = image.yRange * (float(win::width) / win::height);
      glViewport(0, 0, win::width, win::height);

      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, win::width, win::height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, size_t(win::width) * win::height * 4, nullptr, GL_DYNAMIC_DRAW);

      UnregisterPixelBuffer();
      registerPixelBuffer(pbo);

      flags &= ~Resize;
    }
  
    if ((flags & Render) == Render) {
      RenderMandelbrot(image, tex, pbo);
      flags &= ~Render;
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    // std::cout << "Frame time: " << elapsed.count() << " ms" << std::endl;
    
    SDL_GL_SwapWindow(window);
  }

  glDeleteTextures(1, &tex);
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  SDL_GL_DestroyContext(glContext);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
