#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <gaussian.hpp>

// Simple struct to hold rotation angles
struct RotationParams {
    float rotationX;
    float rotationY;
    float3 center;
};

// GPU kernel that rotates each vertex around (center.x, center.y, center.z)
// by rotationY around the Y-axis, then rotationX around the X-axis:
__global__
void rotateVerticesKernel(const float3* d_inVertices,
                          float3*       d_outVertices,
                          int           numVertices,
                          Gaussian3D*   d_splats,
                          RotationParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    // Read inputs
    float3 v = d_inVertices[idx];

    // 1) Translate so that 'center' is at (0,0,0)
    v.x -= params.center.x;
    v.y -= params.center.y;
    v.z -= params.center.z;

    // 2) Rotate around the Y-axis by rotationY
    //    Standard rotation Y formula:
    //      newX = cos(y)*x + sin(y)*z
    //      newZ = -sin(y)*x + cos(y)*z
    float cosY = cosf(params.rotationY);
    float sinY = sinf(params.rotationY);
    float xNew = cosY * v.x + sinY * v.z;
    float zNew = -sinY * v.x + cosY * v.z;
    v.x = xNew;
    v.z = zNew;

    // 3) Rotate around the X-axis by rotationX
    //    Standard rotation X formula:
    //      newY = cos(x)*y - sin(x)*z
    //      newZ = sin(x)*y + cos(x)*z
    float cosX = cosf(params.rotationX);
    float sinX = sinf(params.rotationX);
    float yNew = cosX * v.y - sinX * v.z;
    zNew       = sinX * v.y + cosX * v.z;
    v.y = yNew;
    v.z = zNew;

    // 4) Translate back from the origin to 'center'
    v.x += params.center.x;
    v.y += params.center.y;
    v.z += params.center.z;

    // Write back to output array
    d_outVertices[idx] = v;

    // Update splat positions
    Gaussian3D splat = d_splats[idx];
    splat.position = v;
    d_splats[idx] = splat;
}

// Example host function that launches the kernel
// d_inVertices, d_outVertices: device pointers to float3 arrays
// rotationX, rotationY: angles in radians
// center: rotation pivot
void rotateVerticesOnGPU(float3* d_inVertices,
                         float3* d_outVertices,
                         int     numVertices,
                         Gaussian3D* d_splats,
                         float   rotationX,
                         float   rotationY,
                         float3  center)
{
    // Create a struct with rotation parameters
    RotationParams params;
    params.rotationX = rotationX;
    params.rotationY = rotationY;
    params.center    = center;

    // Kernel launch:
    int blockSize = 256;
    int gridSize = (numVertices + blockSize - 1) / blockSize;

    rotateVerticesKernel<<<gridSize, blockSize>>>(d_inVertices,
                                                  d_outVertices,
                                                  numVertices,
                                                  d_splats,
                                                  params);
    cudaDeviceSynchronize(); // or check errors
}
