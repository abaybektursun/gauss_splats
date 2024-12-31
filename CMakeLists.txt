cmake_minimum_required(VERSION 3.10)
project(HelloCUDA LANGUAGES CXX CUDA)

# Enable CUDA language
enable_language(CUDA)

# Create an executable
add_executable(hello_cuda src/hello_cuda.cu)
