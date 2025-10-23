cmake_minimum_required(VERSION 3.24)
project(mandelbrot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Make MSVC runtime match vcpkg defaults (/MD, /MDd)
if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(OUT_DIR    ${CMAKE_SOURCE_DIR}/bin)
set(SRC_DIR    ${CMAKE_SOURCE_DIR}/src)
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)

add_executable(mandelbrot
  ${SRC_DIR}/main.cpp
  ${SRC_DIR}/mandelbrot.cu
  ${SRC_DIR}/Shaders.cpp
)

target_include_directories(mandelbrot PRIVATE ${SRC_DIR})

# Copy shaders after build
add_custom_command(
  TARGET mandelbrot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${SHADER_DIR}
          $<TARGET_FILE_DIR:mandelbrot>/shaders
  COMMENT "Copying shaders to output directory..."
)

# OpenGL (system)
find_package(OpenGL REQUIRED)
target_link_libraries(mandelbrot PRIVATE OpenGL::GL)

# -------- GLEW --------
if (WIN32)
  # Use vcpkg (CONFIG packages) on Windows; no pkg-config
  find_package(GLEW CONFIG REQUIRED)
  target_link_libraries(mandelbrot PRIVATE GLEW::GLEW)
else()
  # Linux/mac: prefer config, else pkg-config fallback
  find_package(GLEW)
  if (GLEW_FOUND)
    target_link_libraries(mandelbrot PRIVATE GLEW::GLEW)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLEW REQUIRED glew)
    target_include_directories(mandelbrot PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(mandelbrot PRIVATE ${GLEW_LIBRARIES})
  endif()
endif()

# -------- SDL3 --------
if (WIN32)
  # Use vcpkg (CONFIG packages) on Windows; no pkg-config
  find_package(SDL3 CONFIG REQUIRED)
  target_link_libraries(mandelbrot PRIVATE SDL3::SDL3)
else()
  find_package(SDL3)
  if (SDL3_FOUND)
    target_link_libraries(mandelbrot PRIVATE SDL3::SDL3)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL3 REQUIRED sdl3)
    target_include_directories(mandelbrot PRIVATE ${SDL3_INCLUDE_DIRS})
    target_link_libraries(mandelbrot PRIVATE ${SDL3_LIBRARIES})
  endif()
endif()

# Windows niceties (avoid min/max & heavy <Windows.h>)
if (WIN32)
  target_compile_definitions(mandelbrot PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  # If you link shared GLEW/SDL3 via vcpkg, you can copy the DLLs next to the exe:
  add_custom_command(TARGET mandelbrot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:GLEW::GLEW> $<TARGET_FILE_DIR:mandelbrot> 2>NUL || cmd /c exit /b 0)
  add_custom_command(TARGET mandelbrot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:mandelbrot> 2>NUL || cmd /c exit /b 0)
endif()

# CUDA
set_target_properties(mandelbrot PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)
set_source_files_properties(${SRC_DIR}/mandelbrot.cu PROPERTIES LANGUAGE CUDA)
