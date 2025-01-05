#pragma once

#include <cuda_runtime.h>

/**
 * @brief Defines a 3D Gaussian structure.
 */
struct Gaussian3D {
    float3 position;  // (x, y, z) μ: Mean position in 3D
    float3 scale;     // Scaling factors along x, y, z axes
    float4 rotation;  // Rotation represented as a quaternion
    float  opacity;   // α: Opacity value between 0 and 1
    float3 color;     // (r, g, b)
    float  intensity; // Optional: brightness or weight
    // Later: spherical harmonics, etc.
};