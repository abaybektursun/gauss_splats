#pragma once

#include <cuda_runtime.h>
#include "gaussian.hpp"
#include "camera.hpp"
#include <vector> 

/**
 * @brief Holds the result of projecting a Gaussian into 2D.
 */
struct ProjectedSplat {
    int tileID;        // Which tile in tile-based raster
    float depth;       // z-value for sorting
    int pixelX;        // Final 2D pixel coordinate (u)
    int pixelY;        // Final 2D pixel coordinate (v)
    Gaussian3D* gaussian;  // Pointer to the original Gaussian
};

/**
 * @brief GPU kernel to project 3D Gaussians into 2D splats.
 *
 * @param d_gaussians    Device pointer to array of Gaussian3D.
 * @param d_outSplats    Device pointer to an output array of ProjectedSplat.
 * @param numGaussians   Number of Gaussians in the array.
 * @param cam            Camera parameters (orthographic).
 * @param tile_size      Side length in pixels of each tile.
 */
__global__
void projectGaussiansKernel(const Gaussian3D* d_gaussians,
                            ProjectedSplat* d_outSplats,
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
void tiledBlendingKernel(const ProjectedSplat*  d_inSplats,
                         float4*               d_outImage,
                         const int*            d_tileRangeStart,
                         const int*            d_tileRangeEnd,
                         OrthoCameraParams     cam,
                         int                   tile_size);

/**
 * @brief CPU function to compute per-tile start/end indices after sorting splats.
 */
void computeTileRanges(std::vector<ProjectedSplat>& h_sortedSplats,
                       int totalTiles,
                       std::vector<int>& tileRangeStart,
                       std::vector<int>& tileRangeEnd);

