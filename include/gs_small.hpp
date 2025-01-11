
/**
 * @brief Holds the result of projecting a Gaussian into 2D.
 */
struct ProjectedGaussian_small {
    int tileID;        // Which tile in tile-based raster
    float depth;       // z-value for sorting
    int pixelX;        // Final 2D pixel coordinate (u)
    int pixelY;        // Final 2D pixel coordinate (v)
    float2x2 sigma2D;  // Screen-space covariance matrix
    // These are not camera min and max, but the bounding box of the splat
    int2 bboxMin;      // Bounding box min (u_min, v_min)
    int2 bboxMax;      // Bounding box max (u_max, v_max)
    Gaussian3D* gaussian;  // Pointer to the original Gaussian
};

/**
 * @brief GPU kernel to project 3D Gaussians into 2D splats.
 *
 * @param d_gaussians    Device pointer to array of Gaussian3D.
 * @param d_outSplats    Device pointer to an output array of ProjectedGaussian_small.
 * @param numGaussians   Number of Gaussians in the array.
 * @param cam            Camera parameters (orthographic).
 * @param tile_size      Side length in pixels of each tile.
 */
__global__
void projectGaussiansKernel_small(const Gaussian3D* d_gaussians,
                            ProjectedGaussian_small* d_outSplats,
                            int numGaussians,
                            OrthoCameraParams cam,
                            int tile_size = 16);

/**
 * @brief GPU kernel to blend splats tile-by-tile.
 */
__global__
void tiledBlendingKernel_small(const ProjectedGaussian_small*  d_inSplats,
                         float4*               d_outImage,
                         const int*            d_tileRangeStart,
                         const int*            d_tileRangeEnd,
                         OrthoCameraParams     cam,
                         int                   tile_size);


void generateTileRanges_small(
    const ProjectedGaussian_small* d_outSplats,
    int totalTiles,
    int tileSize,
    int totalTileSplats,
    int* d_tileRangeStart,
    int* d_tileRangeEnd);