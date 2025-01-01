#pragma once

#include <cuda_runtime.h>

/**
 * @brief Defines a 3D Gaussian structure.
 */
struct Gaussian3D {
    float3 position;  // (x, y, z)
    float3 scale;     // scale for x, y, z
    float4 rotation;  // quaternion
    float  opacity;   // alpha
    float3 color;     // (r, g, b)
    float  intensity; // brightness multiplier
    // Later: spherical harmonics, etc.
};

