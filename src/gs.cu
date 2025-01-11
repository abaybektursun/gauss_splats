#include <cstdio>  // for printf (if needed)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gs.hpp"



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

// Device functions

/**
 * @brief Computes the eigenvalues and eigenvectors of a 2x2 symmetric matrix.
 *
 * This function computes the eigenvalues and eigenvectors of a symmetric
 * 2x2 matrix using a numerically stable method.
 *
 * The matrix is:
 *   [ a  b ]
 *   [ b  c ]
 *
 * @param a       Element at (0,0) of the matrix.
 * @param b       Element at (0,1) and (1,0) of the matrix.
 * @param c       Element at (1,1) of the matrix.
 * @param lambda1 Output: Largest eigenvalue.
 * @param lambda2 Output: Smallest eigenvalue.
 * @param v1      Output: Eigenvector corresponding to lambda1.
 * @param v2      Output: Eigenvector corresponding to lambda2.
 */
__device__ void computeEigenValuesAndVectors(
    float a, float b, float c,
    float& lambda1, float& lambda2,
    float2& v1, float2& v2)
{
    // Compute the trace and determinant of the matrix
    float trace = a + c;
    float diff = a - c;
    float discriminant = sqrtf(diff * diff + 4.0f * b * b);

    // Compute the eigenvalues
    lambda1 = 0.5f * (trace + discriminant);
    lambda2 = 0.5f * (trace - discriminant);

    // Compute the eigenvectors
    if (b != 0.0f)
    {
        float t = lambda1 - a;
        float norm = hypotf(t, b);
        // avoid division by zero
        if (norm > 0.0f)
        {
            v1 = make_float2(b / norm, t / norm);
        }
        else
        {
            // Default to a unit vector
            v1 = make_float2(1.0f, 0.0f);
        }

        t = lambda2 - a;
        norm = hypotf(b, t);
        v2 = make_float2(t / norm, b / norm);
    }
    else
    {
        // If b is zero, the matrix is diagonal
        v1 = (a >= c) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 1.0f);
        v2 = (a >= c) ? make_float2(0.0f, 1.0f) : make_float2(1.0f, 0.0f);
    }
}

/**
 * @brief Computes the axis-aligned bounding box of an ellipse in screen space.
 *
 * Given the 2D covariance matrix of a splat and its screen position, this function
 * computes the bounding box that fully contains the ellipse representing the splat.
 *
 * @param sigma11   Element (0,0) of the 2D covariance matrix.
 * @param sigma12   Element (0,1) and (1,0) of the 2D covariance matrix.
 * @param sigma22   Element (1,1) of the 2D covariance matrix.
 * @param centerX   Center x-coordinate of the ellipse in pixels.
 * @param centerY   Center y-coordinate of the ellipse in pixels.
 * @param k         Scaling factor (e.g., 2 for 95% confidence interval).
 * @param imageWidth  Width of the screen/image in pixels.
 * @param imageHeight Height of the screen/image in pixels.
 * @param bboxMin   Output: Minimum (x, y) coordinates of the bounding box.
 * @param bboxMax   Output: Maximum (x, y) coordinates of the bounding box.
 */
__device__ void computeSplatBoundingBox(
    float sigma11, float sigma12, float sigma22,
    int centerX, int centerY, float k,
    int imageWidth, int imageHeight,
    int2& bboxMin, int2& bboxMax)
{
    // Compute eigenvalues and eigenvectors of the covariance matrix
    float lambda1, lambda2;
    float2 v1, v2;
    computeEigenValuesAndVectors(sigma11, sigma12, sigma22, lambda1, lambda2, v1, v2);

    // TEMP DEBUG: TODO REMOVE
    if (lambda1 < 0.0f || lambda2 < 0.0f) {
        printf("ERROR: Negative eigenvalues: lambda1=%f, lambda2=%f\n", lambda1, lambda2);
    }
    // Compute radii along the principal axes
    lambda1 = max(lambda1, 0.0f);
    lambda2 = max(lambda2, 0.0f);

    float radius1 = k * sqrtf(lambda1);
    float radius2 = k * sqrtf(lambda2);


    // Eigenvectors components
    float cosTheta = v1.x;
    float sinTheta = v1.y;
    float vecLength = sqrtf(v1.x * v1.x + v1.y * v1.y);
    if (vecLength > 0.0f)
    {
        cosTheta = v1.x / vecLength;
        sinTheta = v1.y / vecLength;
    }
    else
    {
        // Default orientation if eigenvector length is zero
        cosTheta = 1.0f;
        sinTheta = 0.0f;
    }

    // Compute the extents along x and y axes
    float absCosTheta = fabsf(cosTheta);
    float absSinTheta = fabsf(sinTheta);

    // Compute half-widths of the bounding box
    float dx = radius1 * absCosTheta + radius2 * absSinTheta;
    float dy = radius1 * absSinTheta + radius2 * absCosTheta;

    // Compute bounding box coordinates
    int minX = static_cast<int>(floorf(centerX - dx));
    int maxX = static_cast<int>(ceilf(centerX + dx));
    int minY = static_cast<int>(floorf(centerY - dy));
    int maxY = static_cast<int>(ceilf(centerY + dy));

    // Clamp to image boundaries
    minX = max(minX, 0);
    maxX = min(maxX, imageWidth - 1);
    minY = max(minY, 0);
    maxY = min(maxY, imageHeight - 1);

    bboxMin = make_int2(minX, minY);
    bboxMax = make_int2(maxX, maxY);
}

/**
 * Kernel: project 3D Gaussians to 2D splats.
 */
__global__
void projectGaussiansKernel(const Gaussian3D* d_gaussians,
                            ProjectedGaussian* d_outSplats,
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

    float s_x2 = g.scale.x * g.scale.x;
    float s_y2 = g.scale.y * g.scale.y;
    float s_z2 = g.scale.z * g.scale.z;

    // Scale Matrix: S
    // | s_x2  0   0 |
    // | 0   s_y2  0 |
    // | 0    0  s_z2|

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
    float M11 = R11 * s_x2;  float M12 = R12 * s_y2; float M13 = R13 * s_z2;
    float M21 = R21 * s_x2;  float M22 = R22 * s_y2; float M23 = R23 * s_z2;
    float M31 = R31 * s_x2;  float M32 = R32 * s_y2; float M33 = R33 * s_z2;

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

    float ScaleScreenX = (cam.imageWidth - 1) / (cam.xMax - cam.xMin);
    float ScaleScreenY = (cam.imageHeight - 1) / (cam.yMax - cam.yMin);
    
    // Since sigma13 and sigma23 are not involved in u and v, they can be ignored
    float sigma2D_11 = sigma11 * ScaleScreenX * ScaleScreenX;
    float sigma2D_12 = sigma12 * ScaleScreenX * ScaleScreenY;
    float sigma2D_22 = sigma22 * ScaleScreenY * ScaleScreenY;    


    int u, v;
    orthographicProject(x, y, cam, u, v);

    // Check if the projected point is outside the image
    if (u < 0 || u >= cam.imageWidth ||
        v < 0 || v >= cam.imageHeight)
    {
        // TODO: update the renderer to handle invalid splats
        //d_outSplats[idx].tileID = -1;
        d_outSplats[idx].invalid = true;
        return;
    }

    // Determine tile
    int tileX  = u / tile_size;
    int tileY  = v / tile_size;
    int tileID = tileY * (cam.imageWidth / tile_size) + tileX;

    //d_outSplats[idx].tileID  = tileID;
    d_outSplats[idx].depth   = z;
    d_outSplats[idx].pixelX  = u;
    d_outSplats[idx].pixelY  = v;
    d_outSplats[idx].sigma2D.row1 = make_float2(sigma2D_11, sigma2D_12);
    d_outSplats[idx].sigma2D.row2 = make_float2(sigma2D_12, sigma2D_22);
    computeSplatBoundingBox(sigma2D_11, sigma2D_12, sigma2D_22, u, v, 2.0f, cam.imageWidth, cam.imageHeight,
                            d_outSplats[idx].bboxMin, d_outSplats[idx].bboxMax);
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
        //printf("Before blend: dest=(%.3f, %.3f, %.3f, %.3f) src=(%.3f, %.3f, %.3f, %.3f)\n", 
        //       dest.x, dest.y, dest.z, dest.w, src.x, src.y, src.z, src.w);
        dest.x += oneMinusA * alphaSplat * src.x;
        dest.y += oneMinusA * alphaSplat * src.y;
        dest.z += oneMinusA * alphaSplat * src.z;
        dest.w += oneMinusA * alphaSplat;
        //printf("After blend: dest=(%.3f, %.3f, %.3f, %.3f)\n", 
        //       dest.x, dest.y, dest.z, dest.w);
    }
}


/**
 * Kernel: for each tile (block), blend all its splats in thread order.
 */
__global__
void tiledBlendingKernel(const TileSplat*      d_tileSplats,
                         float4*               d_outImage,
                         const int*            d_tileRangeStart,
                         const int*            d_tileRangeEnd,
                         OrthoCameraParams     cam,
                         int                   tile_size)
{
    // TODO: temporary, remove this after tile_size is dynamic
    if (tile_size > 32) {
        printf("ERROR: tile_size max is 32\n");
        return;
    }
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
    __shared__ float4 tilePixels[1024];  // for now max is 32x32
    tilePixels[localIdx] = d_outImage[globalPixelIdx];
    __syncthreads();


    // Loop over splats in this tile
    for (int i = start; i < end; i++) {
        TileSplat s = d_tileSplats[i];
        Gaussian3D* gPtr = s.gaussPtr;
        if (gPtr == nullptr) {
            printf("Thread (%d): gPtr is nullptr\\n", threadIdx.x);
            continue;
        }

        // Check if current pixel lies within the splat's bounding box
        if (globalX >= s.bboxMin.x && globalX <= s.bboxMax.x &&
            globalY >= s.bboxMin.y && globalY <= s.bboxMax.y)
        {
            // Compute delta between pixel and splat center
            float deltaX = globalX - s.pixelX;
            float deltaY = globalY - s.pixelY;
            float2 delta = make_float2(deltaX, deltaY);

            // Compute exponent for Gaussian weight
            float a = s.sigma2D.row1.x;
            float b = s.sigma2D.row1.y; // Same as s.sigma2D.row2.x
            float c = s.sigma2D.row2.y;
            float det = a * c - b * b;
            if (fabs(det) < 1e-6f) {
                //printf("Thread (%d): Determinant too small. det=%f\n", threadIdx.x, det);
                continue; // Skip if determinant is too small
            }

            float invDet = 1.0f / det;
            // Inverse of sigma2D
            float invA =  invDet * c;
            float invB = -invDet * b;
            float invC =  invDet * a;
            // Compute exponent
            float e = -0.5f * (delta.x * (invA * delta.x + invB * delta.y) +
                               delta.y * (invB * delta.x + invC * delta.y));
            // Compute weight
            float weight = expf(e);
            // Skip if weight is negligible
            if (weight < 1e-4f) {
                //printf("Thread (%d): Weight too small. e=%f weight=%f\n", threadIdx.x, e, weight);
                continue;
            }


            // Compute source color with weight
            float alpha = gPtr->opacity * weight;
            float4 srcColor = make_float4(gPtr->color.x * weight,
                                          gPtr->color.y * weight,
                                          gPtr->color.z * weight,
                                          alpha);
            

            //printf("Color: r=%f g=%f b=%f a=%f\n", 
            //srcColor.x, srcColor.y, srcColor.z, srcColor.w);

            // **Add printf to check values before blending**
            //printf("Thread (%d): Blending at pixel (%d,%d) with alpha=%f\n", threadIdx.x, globalX, globalY, alpha);

            // Alpha blend
            alphaBlend(tilePixels[localIdx], srcColor);
        }
    }

    __syncthreads();
    d_outImage[globalPixelIdx] = tilePixels[localIdx];
}


__global__
void scatterTileRanges(const int* uniqueTileIDs,
                       const int* tileStarts,
                       const int* tileEnds,
                       int*       d_tileRangeStart,
                       int*       d_tileRangeEnd,
                       int        numUniqueTiles,
                       int        totalTiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numUniqueTiles) return;

    int tile = uniqueTileIDs[i];
    // Check tile is in range
    if (tile >= 0 && tile < totalTiles)
    {
        d_tileRangeStart[tile] = tileStarts[i];
        // +1 so that tileRangeEnd is "one past" the last index, as in your CPU function
        d_tileRangeEnd[tile]   = tileEnds[i] + 1;
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


void generateTileRanges(
    const TileSplat* d_tileSplats,
    int totalTiles,
    int tileSize,
    int totalTileSplats,
    int* d_tileRangeStart,
    int* d_tileRangeEnd)
{
        // 1) Fill tileRangeStart and tileRangeEnd on the device with -1
        thrust::device_ptr<int> d_startPtr(d_tileRangeStart);
        thrust::device_ptr<int> d_endPtr(d_tileRangeEnd);
        thrust::fill(d_startPtr, d_startPtr + totalTiles, -1);
        thrust::fill(d_endPtr,   d_endPtr + totalTiles,   -1);

        // 2) Create an array of indices [0, 1, 2, ...] for the splats
        thrust::device_vector<int> d_indices(totalTileSplats);
        thrust::sequence(d_indices.begin(), d_indices.end());

        // 3) Extract tileIDs from the sorted splats
        thrust::device_vector<int> d_tileIDs(totalTileSplats);
        thrust::transform(
            thrust::device_pointer_cast(d_tileSplats),
            thrust::device_pointer_cast(d_tileSplats + totalTileSplats),
            d_tileIDs.begin(),
            [] __device__ (const TileSplat &s) {
                return s.tileID;
            }
        );

        // 4) reduce_by_key for min and max indices
        thrust::device_vector<int> d_tileIDsOut(totalTileSplats);
        thrust::device_vector<int> d_tileStartsOut(totalTileSplats);
        thrust::device_vector<int> d_tileEndsOut(totalTileSplats);

        // (a) find the FIRST index for each tile
        auto min_end = thrust::reduce_by_key(
            d_tileIDs.begin(), d_tileIDs.end(),  // keys
            d_indices.begin(),                   // values
            d_tileIDsOut.begin(),                // output keys
            d_tileStartsOut.begin(),             // output values (min indices)
            thrust::equal_to<int>(),
            thrust::minimum<int>()
        );

        // (b) find the LAST index for each tile
        auto max_end = thrust::reduce_by_key(
            d_tileIDs.begin(), d_tileIDs.end(),
            d_indices.begin(),
            thrust::make_discard_iterator(),    // we don't need to store keys again
            d_tileEndsOut.begin(),
            thrust::equal_to<int>(),
            thrust::maximum<int>()
        );

        // how many unique tiles did we actually get?
        int numUniqueTiles = static_cast<int>(min_end.first - d_tileIDsOut.begin());

        // 5) Scatter results directly on the GPU
        // We'll launch a kernel to write tileRangeStart[tile], tileRangeEnd[tile].
        {
            int blockSize = tileSize*tileSize;
            int gridSize = (numUniqueTiles + blockSize - 1) / blockSize;
            scatterTileRanges<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(d_tileIDsOut.data()),
                thrust::raw_pointer_cast(d_tileStartsOut.data()),
                thrust::raw_pointer_cast(d_tileEndsOut.data()),
                d_tileRangeStart,
                d_tileRangeEnd,
                numUniqueTiles,
                totalTiles
            );
            cudaDeviceSynchronize();
        }
}


// Sort d_tileSplats by tileID and depth
void sortTileSplats(const GPUData& gpuData, int totalTileSplats){
        thrust::device_vector<unsigned long long> d_keys(totalTileSplats);
        thrust::transform(
            thrust::device_pointer_cast(gpuData.d_tileSplats),
            thrust::device_pointer_cast(gpuData.d_tileSplats + totalTileSplats),
            d_keys.begin(),
            [] __device__ (const TileSplat& s) {
                return packTileDepth(s.tileID, s.depth);
            }
        );
        thrust::device_ptr<TileSplat> d_splats_ptr(gpuData.d_tileSplats);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_splats_ptr);
}


__global__
void countTilesKernel(const ProjectedGaussian* d_splats,
                      int vertexCount,
                      int tileSize,
                      int tilesInX,
                      int tilesInY,
                      int* d_splatCounts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertexCount) return;
    
    // 1) Read one ProjectedGaussian from the array
    ProjectedGaussian s = d_splats[idx];
    
    // 2) Convert pixel-space bounding box to tile indices
    //    e.g. if bboxMin.x=50 and tileSize=16 => tileMinX=3
    int tileMinX = s.bboxMin.x / tileSize;
    int tileMaxX = s.bboxMax.x / tileSize;
    int tileMinY = s.bboxMin.y / tileSize;
    int tileMaxY = s.bboxMax.y / tileSize;

    // 3) Clamp tile indices so they don’t go out of [0..tilesInX-1], [0..tilesInY-1]
    tileMinX = max(0, tileMinX);
    tileMaxX = min(tileMaxX, tilesInX - 1);
    tileMinY = max(0, tileMinY);
    tileMaxY = min(tileMaxY, tilesInY - 1);

    // 4) Count how many tiles this splat covers
    //    e.g. if tileMinX=3 and tileMaxX=4 => coverage in X is (4-3+1)=2 tiles
    //    similarly for Y, then multiply X coverage * Y coverage
    int count = (tileMaxX - tileMinX + 1) * (tileMaxY - tileMinY + 1);

    // 5) Write that count to the d_splatCounts array.
    d_splatCounts[idx] = count;
}


__global__
void expandTilesKernel(const ProjectedGaussian* d_splats,
                      int* d_splatOffsets,
                      TileSplat* d_tileSplats,
                      int vertexCount,
                      int tileSize,
                      int tilesInX,
                      int tilesInY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertexCount) return;
    
    // 1) Read one ProjectedGaussian from the array
    ProjectedGaussian s = d_splats[idx];
    
    // 2) Convert pixel-space bounding box to tile indices
    //    e.g. if bboxMin.x=50 and tileSize=16 => tileMinX=3
    int tileMinX = s.bboxMin.x / tileSize;
    int tileMaxX = s.bboxMax.x / tileSize;
    int tileMinY = s.bboxMin.y / tileSize;
    int tileMaxY = s.bboxMax.y / tileSize;

    // 3) Clamp tile indices so they don’t go out of [0..tilesInX-1], [0..tilesInY-1]
    tileMinX = max(0, tileMinX);
    tileMaxX = min(tileMaxX, tilesInX - 1);
    tileMinY = max(0, tileMinY);
    tileMaxY = min(tileMaxY, tilesInY - 1);

    int start = d_splatOffsets[idx];
    
    int localIndex = 0;
    for (int ty = tileMinY; ty <= tileMaxY; ty++) {
        for (int tx = tileMinX; tx <= tileMaxX; tx++) {
            int tileID = ty * tilesInX + tx;
            // fill one TileSplat entry
            TileSplat spl;
            spl.tileID = tileID;
            spl.depth    = d_splats[idx].depth;
            spl.sigma2D  = d_splats[idx].sigma2D;
            spl.pixelX   = d_splats[idx].pixelX;
            spl.pixelY   = d_splats[idx].pixelY;
            spl.bboxMin  = d_splats[idx].bboxMin;  // or tile-clipped
            spl.bboxMax  = d_splats[idx].bboxMax;
            spl.gaussPtr = d_splats[idx].gaussian;

            // write to d_tileSplats[start + localIndex];
            // localIndex increments from 0..(count-1)
            d_tileSplats[start + localIndex] = spl;
            localIndex++;
        }
    }

}