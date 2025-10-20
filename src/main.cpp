#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_video.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include <iostream>

#include "../include/mandelbrot.cuh"

namespace gl {
  extern const int kVersionMajor = 3;
  extern const int kVersionMinor = 3;
};

namespace win {
  int width = 1920;
  int height = 1080;
};

static const char* vertexSource = 
R"(#version 330 core

void main() {
  const vec2 pos[3] = vec2[3](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
  );

  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
})";

static const char* fragmentSource = 
R"(#version 330 core

out vec4 fragColor;
  
void main() {
  fragColor = vec4(0.5, 0.7, 1.0, 1.0);
})";

int main() {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::cerr << "Failed to initialize SDL" << std::endl;
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, gl::kVersionMajor);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, gl::kVersionMinor);
  
  SDL_Window *window = SDL_CreateWindow("Mandelbrot", win::width, win::height, SDL_WINDOW_OPENGL);
  if (!window) {
    std::cerr << "Failed to create window" << std::endl;
    return -1;
  }
  auto glContext = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, glContext);

  glewExperimental = GL_TRUE;
  if(glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexSource, nullptr); 
  glCompileShader(vertexShader);

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
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

  bool isRunning = true;
  while (isRunning){
    SDL_Event event;
    while (SDL_PollEvent(&event)  ) {
      switch (event.type) {
        case SDL_EVENT_KEY_DOWN:
          switch (event.key.scancode) {
            case SDL_SCANCODE_ESCAPE:
              isRunning = false;
              break;
            default:
              continue;
          }
        case SDL_EVENT_QUIT:
          isRunning = false;
          break;
      } 
    }

    glUseProgram(program);

    glViewport(0, 0, win::width, win::height);
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    SDL_GL_SwapWindow(window);
  }

  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  SDL_GL_DestroyContext(glContext);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
