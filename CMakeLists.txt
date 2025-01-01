cmake_minimum_required(VERSION 3.15)
project(GaussianSplats CUDA CXX)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
# Thrust is included with CUDA

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Add library targets
add_library(gaussian_lib
    src/gaussian.cu
    src/render_utils.cu
    src/camera.cpp
    src/ply_loader.cpp
)

# Add executable
add_executable(gaussian_splats src/main.cu)
target_link_libraries(gaussian_splats PRIVATE gaussian_lib)
