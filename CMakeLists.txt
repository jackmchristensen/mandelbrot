cmake_minimum_required(VERSION 3.24)
project(mandelbrot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(OUT_DIR ${CMAKE_SOURCE_DIR}/bin)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)

add_executable(mandelbrot
  ${SRC_DIR}/main.cpp
  ${SRC_DIR}/mandelbrot.cu
  ${SRC_DIR}/Shaders.cpp
)

target_include_directories(mandelbrot PRIVATE ${SRC_DIR})

# Copy shaders to output folder after build
add_custom_command(
    TARGET mandelbrot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_DIR}
        ${OUT_DIR}/shaders
    COMMENT "Copying shaders to output directory..."
)

find_package(OpenGL REQUIRED)
target_link_libraries(mandelbrot PRIVATE OpenGL::GL)

find_package(GLEW)
if(GLEW_FOUND)
  target_link_libraries(mandelbrot PRIVATE GLEW::GLEW)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLEW REQUIRED glew)
  target_include_directories(mandelbrot PRIVATE ${GLEW_INCLUDE_DIRECTORIES})
  target_link_libraries(mandelbrot PRIVATE ${GLEW_LIBRARIES})
endif()

find_package(SDL3)
if(SDL3_FOUND)
  target_link_libraries(mandelbrot PRIVATE SDL3::SDL3)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SDL3 REQUIRED sdl3)
  target_include_directories(mandelbrot PRIVATE ${SDL3_INCLUDE_DIRECTORIES})
  target_link_libraries(mandelbrot PRIVATE ${SDL3_LIBRARIES})
endif()

set_target_properties(mandelbrot PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR} CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(${SRC_DIR}/mandelbrot.cu PROPERTIES LANGUAGE CUDA)
