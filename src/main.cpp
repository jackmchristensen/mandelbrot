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
  int width = 1280;
  int height = 720;
};

static const char* vertexSource = 
R"(#version 330 core

out vec2 v_uv;

void main() {
  const vec2 pos[3] = vec2[3](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
  );

  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
  v_uv = 0.5 * (pos[gl_VertexID] + 1.0);
})";

static const char* fragmentSource = 
R"(#version 330 core

uniform sampler2D u_tex;

in vec2 v_uv;

out vec4 fragColor;
  
void main() {
  fragColor = texture(u_tex, v_uv);
  // fragColor = vec4(v_uv.x, v_uv.y, 0.0, 1.0f);
})";

int main() {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::cerr << "Failed to initialize SDL" << std::endl;
    return -1;
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, gl::kVersionMajor);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, gl::kVersionMinor);
  
  SDL_Window *window = SDL_CreateWindow("Mandelbrot", win::width, win::height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
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

  auto vendor   = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
  auto renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  std::cout << "GL_VENDOR=" << (vendor?vendor:"?") 
          << "  GL_RENDERER=" << (renderer?renderer:"?") << "\n";

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

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, win::width, win::height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, win::width * win::height * 4, nullptr, GL_DYNAMIC_DRAW);

  registerPixelBuffer(pbo);

  glUseProgram(program);
  GLint loc = glGetUniformLocation(program, "u_tex");
  if (loc == -1) std::cerr << "u_tex not found" << std::endl;
  glUniform1i(loc, 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex);

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

    drawGradient(win::width, win::height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win::width, win::height, GL_RGBA, GL_UNSIGNED_BYTE, reinterpret_cast<const void*>(0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    SDL_GL_SwapWindow(window);
  }

  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  SDL_GL_DestroyContext(glContext);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
