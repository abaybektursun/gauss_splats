#include <cstdio>  // for printf (if needed)
#include "render_utils.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

/**
 * Device-side function to do orthographic projection.
 */
__device__ void orthographicProject(float x, float y,
                                    const OrthoCameraParams& cam,
                                    int& outU, int& outV)
{
    float normalizedX = (x - cam.xMin) / (cam.xMax - cam.xMin);
    float normalizedY = (y - cam.yMin) / (cam.yMax - cam.yMin);

    outU = static_cast<int>(normalizedX * (cam.imageWidth  - 1));
    outV = static_cast<int>(normalizedY * (cam.imageHeight - 1));
}

/**
 * Kernel: project 3D Gaussians to 2D splats.
 */
__global__
void projectGaussiansKernel(const Gaussian3D* d_gaussians,
                            ProjectedSplat* d_outSplats,
                            int numGaussians,
                            OrthoCameraParams cam,
                            int tile_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians) return;

    float x = d_gaussians[idx].position.x;
    float y = d_gaussians[idx].position.y;
    float z = d_gaussians[idx].position.z;

    int u, v;
    orthographicProject(x, y, cam, u, v);

    if (u < 0 || u >= cam.imageWidth ||
        v < 0 || v >= cam.imageHeight)
    {
        d_outSplats[idx].tileID = -1;
        return;
    }

    // Determine tile
    int tileX  = u / tile_size;
    int tileY  = v / tile_size;
    int tileID = tileY * (cam.imageWidth / tile_size) + tileX;

    d_outSplats[idx].tileID  = tileID;
    d_outSplats[idx].depth   = z;
    d_outSplats[idx].pixelX  = u;
    d_outSplats[idx].pixelY  = v;
    // Pointer to the original Gaussian
    d_outSplats[idx].gaussian = (Gaussian3D*)&d_gaussians[idx];
}

/**
 * Device-side inline function for alpha blending.
 */
__device__ inline
void alphaBlend(float4& dest, const float4& src)
{
    float alphaAccum = dest.w;
    float alphaSplat = src.w;

    if (alphaAccum < 1.0f && alphaSplat > 0.0f) {
        float oneMinusA = 1.0f - alphaAccum;
        dest.x += oneMinusA * alphaSplat * src.x;
        dest.y += oneMinusA * alphaSplat * src.y;
        dest.z += oneMinusA * alphaSplat * src.z;
        dest.w += oneMinusA * alphaSplat;
    }
}

/**
 * Kernel: for each tile (block), blend all its splats in thread order.
 */
__global__
void tiledBlendingKernel(const ProjectedSplat*  d_inSplats,
                         float4*               d_outImage,
                         const int*            d_tileRangeStart,
                         const int*            d_tileRangeEnd,
                         OrthoCameraParams     cam,
                         int                   tile_size)
{
    int tileIndex = blockIdx.x;
    int start = d_tileRangeStart[tileIndex];
    int end   = d_tileRangeEnd[tileIndex];
    if (start >= end) return;

    int tilesInX = cam.imageWidth / tile_size;
    int tileX    = tileIndex % tilesInX;
    int tileY    = tileIndex / tilesInX;

    int tileOriginX = tileX * tile_size;
    int tileOriginY = tileY * tile_size;

    int localIdx = threadIdx.x;
    if (localIdx >= tile_size * tile_size) return;

    int localY = localIdx / tile_size;
    int localX = localIdx % tile_size;
    int globalX = tileOriginX + localX;
    int globalY = tileOriginY + localY;
    int globalPixelIdx = globalY * cam.imageWidth + globalX;

    __shared__ float4 tilePixels[256];  // tile_size=16 => 16*16=256
    tilePixels[localIdx] = d_outImage[globalPixelIdx];
    __syncthreads();

    for (int i = start; i < end; i++) {
        ProjectedSplat s = d_inSplats[i];
        if (s.pixelX == globalX && s.pixelY == globalY) {
            Gaussian3D* gPtr = s.gaussian;
            float4 srcColor;
            srcColor.x = gPtr->color.x;
            srcColor.y = gPtr->color.y;
            srcColor.z = gPtr->color.z;
            srcColor.w = gPtr->opacity;

            alphaBlend(tilePixels[localIdx], srcColor);

            if (tilePixels[localIdx].w > 0.999f) {
                break; // early-out
            }
        }
    }

    __syncthreads();
    d_outImage[globalPixelIdx] = tilePixels[localIdx];
}

/**
 * CPU utility: compute tileRangeStart / tileRangeEnd from sorted splats on the host.
 */
void computeTileRanges(std::vector<ProjectedSplat>& h_sortedSplats,
                       int totalTiles,
                       std::vector<int>& tileRangeStart,
                       std::vector<int>& tileRangeEnd)
{
    std::fill(tileRangeStart.begin(), tileRangeStart.end(), -1);
    std::fill(tileRangeEnd.begin(),   tileRangeEnd.end(),   -1);

    if (h_sortedSplats.empty()) return;

    int currentTile = h_sortedSplats[0].tileID;
    if (currentTile >= 0 && currentTile < totalTiles) {
        tileRangeStart[currentTile] = 0;
    }

    for (int i = 1; i < (int)h_sortedSplats.size(); i++) {
        int prevTile = h_sortedSplats[i-1].tileID;
        int thisTile = h_sortedSplats[i].tileID;
        if (thisTile != prevTile) {
            if (prevTile >= 0 && prevTile < totalTiles) {
                tileRangeEnd[prevTile] = i;
            }
            if (thisTile >= 0 && thisTile < totalTiles &&
                tileRangeStart[thisTile] == -1)
            {
                tileRangeStart[thisTile] = i;
            }
        }
    }

    int lastTile = h_sortedSplats.back().tileID;
    if (lastTile >= 0 && lastTile < totalTiles) {
        tileRangeEnd[lastTile] = (int)h_sortedSplats.size();
    }
}
