cmake_minimum_required(VERSION 3.15)
project(GaussianSplats CUDA CXX)

# -------------------------------------------------------
# CUDA settings
# -------------------------------------------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# If you want better debug info, you could add the following:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -G")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g")

# -------------------------------------------------------
# Find packages
# -------------------------------------------------------
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)  # or use a FindSDL2.cmake script

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Thrust typically comes with CUDA, so no special find_package needed for Thrust.

# -------------------------------------------------------
# Include directories
# -------------------------------------------------------
# (Adjust or remove if you have different folder structure)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# -------------------------------------------------------
# Add the main library
# -------------------------------------------------------
# Gather all modular .cpp/.cu files here. 
# Replace or add the names of your newly split files:
#   - gpu_renderer.cu      (if you refactored gaussian.cu + render_utils.cu + camera_gpu.cu)
#   - geometry_utils.cpp   (if you split out bounding sphere / transforms)
#   - camera.cpp
#   - tinyply.cpp
#   - fps_counter.cpp      (if you created a separate .cpp)
#
# If you still have the old files (gaussian.cu, render_utils.cu, camera_gpu.cu), 
# keep them or replace them with the new ones.

add_library(gaussian_lib
    # GPU-related:
    src/gaussian.cu
    src/render_utils.cu
    src/camera_gpu.cu

    # CPU-related:
    src/camera.cpp
    src/tinyply.cpp

    src/ply_utils.cu
    src/sdl_utils.cu

)

# -------------------------------------------------------
# Build the interactive viewer executable
# -------------------------------------------------------
# If you're now calling your main file "main_interactive_splats.cu", keep it here:
# Or if you renamed it to "main.cpp", update accordingly.

add_executable(gaussian_splats src/main_interactive_splats.cu)
target_link_libraries(gaussian_splats
    PRIVATE 
        gaussian_lib
        OpenGL::GL        # from find_package(OpenGL)
        SDL2::SDL2        # from find_package(SDL2)
)

# -------------------------------------------------------
# Build the training executable
# -------------------------------------------------------
# If you also have a "main_train.cu" or "main_train.cpp" for training:
add_executable(gaussian_train src/main_train.cu)
target_link_libraries(gaussian_train
    PRIVATE
        gaussian_lib
        ${OpenCV_LIBS}    # from find_package(OpenCV)
)

