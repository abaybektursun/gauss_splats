#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <iostream>

/**
 * @brief A simple CPU helper function to measure time of a given lambda or functor.
 */
inline void measureTime(const std::string& label, const std::function<void()>& run) {
    auto start = std::chrono::high_resolution_clock::now();
    run();
    auto end   = std::chrono::high_resolution_clock::now();
    auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << label << " took " << ms << " ms\n";
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
