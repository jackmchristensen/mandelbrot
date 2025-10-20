package main

import "core:fmt"

import SDL "vendor:sdl3"
import gl "vendor:OpenGL"

GL_VERSION_MAJOR :: 3 
GL_VERSION_MINOR :: 3 

main :: proc() {
  WINDOW_WIDTH :: 1920
  WINDOW_HEIGHT :: 1080

  init_ok := SDL.Init({.VIDEO})
  if !init_ok {
    fmt.eprintln("Failed to initialize SDL")
    return
  }
  defer SDL.Quit()

  window := SDL.CreateWindow("Mandelbrot", WINDOW_WIDTH, WINDOW_HEIGHT, {.OPENGL})
  if window == nil {
    fmt.eprintln("Failed to create window")
    return
  }
  defer SDL.DestroyWindow(window)

  SDL.GL_SetAttribute(.CONTEXT_PROFILE_MASK, i32(SDL.GLProfile.CORE))
  SDL.GL_SetAttribute(.CONTEXT_MAJOR_VERSION, GL_VERSION_MAJOR)
  SDL.GL_SetAttribute(.CONTEXT_MINOR_VERSION, GL_VERSION_MINOR)

  gl_context := SDL.GL_CreateContext(window)
  defer SDL.GL_DestroyContext(gl_context)

  gl.load_up_to(GL_VERSION_MAJOR, GL_VERSION_MINOR, SDL.gl_set_proc_address)

  program,program_ok := gl.load_shaders_source(vertex_source, fragment_source)
  if !program_ok {
    fmt.eprintln("Failed to create GLSL program")
    return
  }
  defer gl.DeleteProgram(program)

  uniforms := gl.get_uniforms_from_program(program)
  defer delete(uniforms)

  vao: u32
  gl.GenVertexArrays(1, &vao); defer gl.DeleteVertexArrays(1, &vao)
  gl.BindVertexArray(vao)
  
  renderLoop: for {
    event: SDL.Event
    for SDL.PollEvent(&event) {
      #partial switch event.type {
      case .KEY_DOWN:
        #partial switch event.key.scancode {
        case .ESCAPE:
          break renderLoop
        }
      case .QUIT:
        break renderLoop
      }
    }

    gl.UseProgram(program)

    gl.Viewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    gl.ClearColor(0.5, 0.7, 1.0, 1.0)
    gl.Clear(gl.COLOR_BUFFER_BIT)

    gl.DrawArrays(gl.TRIANGLES, 0, 3)

    SDL.GL_SwapWindow(window)
  }
}



vertex_source := `#version 330 core

void main() {
  const vec2 pos[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
  );

  gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}
`

fragment_source := `#version 330 core

out vec4 fragColor;

void main() {
  fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`
