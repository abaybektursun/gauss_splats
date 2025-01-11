
void generateTileRanges_small(
    const ProjectedGaussian_small* d_outSplats,
    int totalTiles,
    int tileSize,
    int vertexCount,
    int* d_tileRangeStart,
    int* d_tileRangeEnd)
{
        // 1) Fill tileRangeStart and tileRangeEnd on the device with -1
        thrust::device_ptr<int> d_startPtr(d_tileRangeStart);
        thrust::device_ptr<int> d_endPtr(d_tileRangeEnd);
        thrust::fill(d_startPtr, d_startPtr + totalTiles, -1);
        thrust::fill(d_endPtr,   d_endPtr + totalTiles,   -1);

        // 2) Create an array of indices [0, 1, 2, ...] for the splats
        thrust::device_vector<int> d_indices(vertexCount);
        thrust::sequence(d_indices.begin(), d_indices.end());

        // 3) Extract tileIDs from the sorted splats
        thrust::device_vector<int> d_tileIDs(vertexCount);
        thrust::transform(
            thrust::device_pointer_cast(d_outSplats),
            thrust::device_pointer_cast(d_outSplats + vertexCount),
            d_tileIDs.begin(),
            [] __device__ (const ProjectedGaussian_small &s) {
                return s.tileID;
            }
        );

        // 4) reduce_by_key for min and max indices
        thrust::device_vector<int> d_tileIDsOut(vertexCount);
        thrust::device_vector<int> d_tileStartsOut(vertexCount);
        thrust::device_vector<int> d_tileEndsOut(vertexCount);

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

/**
 * Kernel: for each tile (block), blend all its splats in thread order.
 */
__global__
void tiledBlendingKernel_small(const ProjectedGaussian_small* d_inSplats,
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
        ProjectedGaussian_small s = d_inSplats[i];
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
 * Kernel: project 3D Gaussians to 2D splats.
 */
__global__
void projectGaussiansKernel_small(const Gaussian3D* d_gaussians,
                            ProjectedGaussian_small* d_outSplats,
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