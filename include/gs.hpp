#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector> 
#include <cmath>

#include "camera.hpp"
#include "cuda_helpers.hpp"

struct GPUData;

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

struct Gaussian3DGrad {
    float3 dposition;  // (x, y, z) μ: Mean position in 3D
    float3 dscale;     // Scaling factors along x, y, z axes
    float4 drotation;  // Rotation represented as a quaternion
    float  dopacity;   // α: Opacity value between 0 and 1
    float3 dcolor;     // (r, g, b)
    float  dintensity; // Optional: brightness or weight
    // Later: spherical harmonics, etc.
};

/**
 * @brief Converts a positive float to its raw 32-bit representation.
 *        For non-negative floats, ascending numeric order = ascending bit order,
 *        so we can compare them directly as unsigned ints.
 */
__device__ inline uint32_t floatBitsForSorting(float depth)
{
    // This union trick gives us the exact bits of the float without reinterpret_cast
    union {
        float     f;
        uint32_t  u;
    } caster;
    caster.f = depth;
    return caster.u;
}


/**
 * @brief Packs tileID (top 32 bits) and depth (bottom 32 bits) into a 64-bit key.
 * 
 * High 32 bits: tileID
 * Low  32 bits: bits of the float depth
 *
 * Sorting in ascending order by this 64-bit key will:
 *   1) First compare tileIDs; smaller tileIDs come first.
 *   2) If tileIDs are equal, then compare the float bits of depth;
 *      smaller depths come first (assuming depth >= 0).
 *
 * @param tileID An integer tile index. (Assumed >= 0 and < 2^31).
 * @param depth  A non-negative float depth value.
 * @return 64-bit key suitable for thrust::sort.
 */
__device__ inline unsigned long long packTileDepth(int tileID, float depth)
{
    // Convert tileID to 64 bits and shift it up by 32 bits
    unsigned long long tilePart  = static_cast<unsigned long long>(tileID) << 32;

    // Convert depth to an unsigned 32-bit pattern
    //  (works well for non-negative depths)
    unsigned long long depthPart = static_cast<unsigned long long>(floatBitsForSorting(depth));

    // Combine them
    return (tilePart | depthPart);
}

struct float2x2 {
    float2 row1;
    float2 row2;
};


struct ProjectedGaussian {
    bool invalid=false;      // True if the splat is not visible
    float depth;       // z-value for sorting
    int pixelX;        // Final 2D pixel coordinate (u)
    int pixelY;        // Final 2D pixel coordinate (v)
    float2x2 sigma2D;  // Screen-space covariance matrix
    // These are not camera min and max, but the bounding box of the splat
    int2 bboxMin;      // Bounding box min (u_min, v_min)
    int2 bboxMax;      // Bounding box max (u_max, v_max)
    Gaussian3D* gaussian;  // Pointer to the original Gaussian
};


struct TileSplat {
    int   tileID;
    float depth;
    float2x2 sigma2D;
    int2  bboxMin, bboxMax;  
    int   pixelX, pixelY;
    Gaussian3D* gaussPtr;
};


__global__
void projectGaussiansKernel(const Gaussian3D* d_gaussians,
                            ProjectedGaussian* d_outSplats,
                            int numGaussians,
                            OrthoCameraParams cam,
                            int tile_size = 16);

/**
 * @brief Device function that performs alpha blending of a splat into the dest color.
 */
__device__ inline
void alphaBlend(float4& dest, const float4& src);

/**
 * @brief GPU kernel to blend splats tile-by-tile.
 */
__global__
void tiledBlendingKernel(const TileSplat*      d_tileSplats,
                         float4*               d_outImage,
                         const int*            d_tileRangeStart,
                         const int*            d_tileRangeEnd,
                         OrthoCameraParams     cam,
                         int                   tile_size);


__global__
void scatterTileRanges(const int* uniqueTileIDs,
                       const int* tileStarts,
                       const int* tileEnds,
                       int*       d_tileRangeStart,
                       int*       d_tileRangeEnd,
                       int        numUniqueTiles,
                       int        totalTiles);

/**
 * @brief Updates the camera parameters based on the orbit angle.
 *
 * @param angleZ The rotation angle around the Z-axis.
 * @param camera The orthographic camera parameters to update.
 * @param sceneMin The minimum bounds of the scene.
 * @param sceneMax The maximum bounds of the scene.
 */
void orbitCamera(float angleZ, OrthoCameraParams& camera, const float3& sceneMin, const float3& sceneMax);



void generateTileRanges(
    const TileSplat* d_tileSplats,
    int totalTiles,
    int tileSize,
    int totalTileSplats,
    int* d_tileRangeStart,
    int* d_tileRangeEnd
);


void sortTileSplats(const GPUData& gpuData, int vertexCount);
__global__
void countTilesKernel(const ProjectedGaussian* d_splats,
                      int vertexCount,
                      int tileSize,
                      int tilesInX,
                      int tilesInY,
                      int* d_splatCounts);
                      
__global__
void expandTilesKernel(const ProjectedGaussian* d_splats,
                      int* d_splatOffsets,
                      TileSplat* d_tileSplats,
                      int vertexCount,
                      int tileSize,
                      int tilesInX,
                      int tilesInY);
