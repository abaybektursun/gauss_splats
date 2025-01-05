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

    Gaussian3D g = d_gaussians[idx];

    float x = g.position.x;
    float y = g.position.y;
    float z = g.position.z;

    float s_x = g.scale.x;
    float s_y = g.scale.y;
    float s_z = g.scale.z;

    // Scale Matrix: S
    // | s_x  0   0 |
    // | 0   s_y  0 |
    // | 0    0  s_z|

    // Convert Quaternion to Rotation Matrix: R
    // | 1-2y^2-2z^2  2xy-2zw      2xz+2yw     |
    // | 2xy+2zw      1-2x^2-2z^2  2yz-2xw     |
    // | 2xz-2yw      2yz+2xw      1-2x^2-2y^2 |

    float  R11 = 1 - 2 * g.rotation.y * g.rotation.y - 2 * g.rotation.z * g.rotation.z;
    float  R12 = 2 * g.rotation.x * g.rotation.y - 2 * g.rotation.z * g.rotation.w;
    float  R13 = 2 * g.rotation.x * g.rotation.z + 2 * g.rotation.y * g.rotation.w;
    float  R21 = 2 * g.rotation.x * g.rotation.y + 2 * g.rotation.z * g.rotation.w;
    float  R22 = 1 - 2 * g.rotation.x * g.rotation.x - 2 * g.rotation.z * g.rotation.z;
    float  R23 = 2 * g.rotation.y * g.rotation.z - 2 * g.rotation.x * g.rotation.w;
    float  R31 = 2 * g.rotation.x * g.rotation.z - 2 * g.rotation.y * g.rotation.w;
    float  R32 = 2 * g.rotation.y * g.rotation.z + 2 * g.rotation.x * g.rotation.w;
    float  R33 = 1 - 2 * g.rotation.x * g.rotation.x - 2 * g.rotation.y * g.rotation.y;

    // Sigma = R * S * R^T = M * R^T; 
    // M = R * S
    float M11 = R11 * s_x;  float M12 = R12 * s_y; float M13 = R13 * s_z;
    float M21 = R21 * s_x;  float M22 = R22 * s_y; float M23 = R23 * s_z;
    float M31 = R31 * s_x;  float M32 = R32 * s_y; float M33 = R33 * s_z;

    // Sigma = M * R^T
    // R^T = | R11 R21 R31 |
    //       | R12 R22 R32 |
    //       | R13 R23 R33 |

    float sigma11 = M11 * R11 + M12 * R12 + M13 * R13;
    float sigma12 = M11 * R21 + M12 * R22 + M13 * R23;
    float sigma13 = M11 * R31 + M12 * R32 + M13 * R33;
    float sigma21 = M21 * R11 + M22 * R12 + M23 * R13;
    float sigma22 = M21 * R21 + M22 * R22 + M23 * R23;
    float sigma23 = M21 * R31 + M22 * R32 + M23 * R33;
    float sigma31 = M31 * R11 + M32 * R12 + M33 * R13;
    float sigma32 = M31 * R21 + M32 * R22 + M33 * R23;
    float sigma33 = M31 * R31 + M32 * R32 + M33 * R33;
    
    // Screen-space covariance matrix
    // | s_x^2*sigma11  s_x*s_y*sigma12 |
    // | s_x*s_y*sigma12 s_y^2*sigma22  |
    float screenSigma11 = s_x * s_x * sigma11; float screenSigma12 = s_x * s_y * sigma12;
    float screenSigma22 = s_y * s_y * sigma22; float screenSigma13 = s_x * s_y * sigma22;

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
void tiledBlendingKernel(const ProjectedSplat* d_inSplats,
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

    /* For 64x64 image and tile_size=16 (64^2/16^2=16):
    0    1   2   3     <- blockIdx.x TileY 0
    4    5   6   7     <- blockIdx.x TileY 1
    8    9  10  11     <- blockIdx.x TileY 2
    12  13  14  15     <- blockIdx.x TileY 3
    */

    int tilesInX = cam.imageWidth / tile_size;
    int tileX    = tileIndex % tilesInX;
    int tileY    = tileIndex / tilesInX;

    // Where does this tile start in the image?
    int tileOriginX = tileX * tile_size;
    int tileOriginY = tileY * tile_size;

    int localIdx = threadIdx.x;
    if (localIdx >= tile_size * tile_size) return;

    // Coordinates in the block
    int localY = localIdx / tile_size;
    int localX = localIdx % tile_size;
    // Coordinates in the image
    int globalX = tileOriginX + localX;
    int globalY = tileOriginY + localY;
    int globalPixelIdx = globalY * cam.imageWidth + globalX;

    // TODO: make this dynamic based on tile_size
    __shared__ float4 tilePixels[256];  // tile_size=16 => 16*16=256
    tilePixels[localIdx] = d_outImage[globalPixelIdx];
    __syncthreads();

    // Blend each splat in range
    for (int i = start; i < end; i++) {
        ProjectedSplat s = d_inSplats[i];
        Gaussian3D* gPtr = s.gaussian;

        // Build the source color
        float4 srcColor = make_float4(gPtr->color.x,
                                      gPtr->color.y,
                                      gPtr->color.z,
                                      gPtr->opacity);

        // === NEW: If globalX, globalY is within +/-1 of s.pixelX, s.pixelY
        //           then alphaBlend. That covers a 3x3 block for each splat.
        //           Increase or decrease this range as you like.
        const int radius = 1; // half-size of your dot
        if (abs(globalX - s.pixelX) <= radius &&
            abs(globalY - s.pixelY) <= radius)
        {
            // Now do your alphaBlend
            alphaBlend(tilePixels[localIdx], srcColor);

            // Optionally break if fully opaque
            if (tilePixels[localIdx].w > 0.999f) {
                break;
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

void orbitCamera(float angleZ, OrthoCameraParams& camera, const float3& sceneMin, const float3& sceneMax)
{
    float cx = 0.5f * (sceneMin.x + sceneMax.x);
    float cy = 0.5f * (sceneMin.y + sceneMax.y);
    float dx = 0.5f * (sceneMax.x - sceneMin.x);
    float dy = 0.5f * (sceneMax.y - sceneMin.y);

    float cosA = cosf(angleZ);
    float sinA = sinf(angleZ);

    // Define the four corners of the bounding box relative to the center
    float corners[4][2] = {
        {-dx, -dy},
        { dx, -dy},
        { dx,  dy},
        {-dx,  dy}
    };

    float minX = 1e30f, maxX = -1e30f;
    float minY = 1e30f, maxY = -1e30f;

    // Rotate each corner and find the new bounding box
    for(int i = 0; i < 4; ++i){
        float x = corners[i][0];
        float y = corners[i][1];
        // Rotate
        float rx = x * cosA - y * sinA;
        float ry = x * sinA + y * cosA;
        // Translate back to original center
        rx += cx;
        ry += cy;
        // Update bounding box
        if(rx < minX) minX = rx;
        if(rx > maxX) maxX = rx;
        if(ry < minY) minY = ry;
        if(ry > maxY) maxY = ry;
    }

    camera.xMin = minX;
    camera.xMax = maxX;
    camera.yMin = minY;
    camera.yMax = maxY;
}