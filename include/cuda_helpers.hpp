#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip> // For setw

/**
 * @brief A simple CPU helper function to measure time with aligned output.
 */
inline void measureTime(const std::string& label, const std::function<void()>& run) {
    auto start = std::chrono::high_resolution_clock::now();
    run();
    auto end   = std::chrono::high_resolution_clock::now();
    auto us    = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Format with aligned columns: <label> <spaces> <time> μs
    const int labelWidth = 30;  // Adjust this value based on your longest label
    const int timeWidth = 10;   // Width for the timing value
    
    std::cout << std::left << std::setw(labelWidth) << label 
              << std::right << std::setw(timeWidth) << std::fixed << std::setprecision(3)
              << (us / 1000.0) << " μs\n";
}

/**
 * @brief Example GPU kernel to sum vertices (float3).
 *
 * @param vertices    Device pointer to float3 array
 * @param result      Device pointer to float[3] (accumulated x,y,z)
 * @param vertexCount Number of vertices
 */
__global__
void sumVertices(float3* vertices, float* result, int vertexCount);

// Optionally add error-check macros or other CUDA helpers here...
// e.g. #define CUDA_CHECK(x) ...
