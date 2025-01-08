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
void tiledBlendingKernel(const ProjectedSplat* d_inSplats,
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
        ProjectedSplat s = d_inSplats[i];
        Gaussian3D* gPtr = s.gaussian;
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