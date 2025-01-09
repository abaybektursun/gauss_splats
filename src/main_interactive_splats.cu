#include <iostream>
#include <random>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>

#include "camera.hpp"
#include "gaussian.hpp"
#include "render_utils.hpp"
#include "cuda_helpers.hpp"
#include "camera_gpu.cu"
#include "ply_utils.cu"
#include "sdl_utils.cu"

const short WINDOW_WIDTH = 512*1.5;
const short WINDOW_HEIGHT = 512*1.5;

int main() {
    SDLApp app(WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!app.init("Gaussian Splats Viewer")) {
        // Handle error (already printed in init)
        return 1;
    }

    // Load PLY file
    std::vector<float3> originalVertices;
    std::vector<float3> originalColors;
    int vertexCount = 0;
    std::vector<Gaussian3D> h_splats(0); // Will be resized in read_init_ply
    read_init_ply(originalVertices, vertexCount, originalColors, h_splats);
    

    // Calculate bounding sphere once
    BoundingSphere boundingSphere = calculateBoundingSphere(originalVertices);
    // Set up fixed camera parameters based on bounding sphere
    OrthoCameraParams camera;
    camera.imageWidth = WINDOW_WIDTH;
    camera.imageHeight = WINDOW_HEIGHT;

    
    // Calculate fixed camera bounds that will work for any rotation
    float aspectRatio = float(camera.imageWidth) / float(camera.imageHeight);
    float margin = 1.0f; // Add some margin around the object
    float radius = boundingSphere.radius * margin;

    // Calculate the half-width and half-height of our view volume
    float halfWidth = radius;
    float halfHeight = radius;
    if (aspectRatio > 1.0f) {
        halfWidth *= aspectRatio;
    } else {
        halfHeight /= aspectRatio;
    }
    
    // Center the camera bounds around the object's center
    camera.xMin = boundingSphere.center.x - halfWidth;
    camera.xMax = boundingSphere.center.x + halfWidth;
    camera.yMin = boundingSphere.center.y - halfHeight;
    camera.yMax = boundingSphere.center.y + halfHeight;


    // Allocate CUDA resources
    float4* d_image = nullptr;
    Gaussian3D* d_splats = nullptr;
    ProjectedSplat* d_outSplats = nullptr;
    float3* d_vertices = nullptr;
    
    cudaMalloc(&d_image, camera.imageWidth * camera.imageHeight * sizeof(float4));
    cudaMalloc(&d_splats, vertexCount * sizeof(Gaussian3D));
    cudaMalloc(&d_outSplats, vertexCount * sizeof(ProjectedSplat));
    cudaMalloc(&d_vertices, vertexCount * sizeof(float3));
    cudaMemcpy(d_vertices, originalVertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);
    float3* d_originalVertices = nullptr;  // Add this with other GPU allocations
    cudaMalloc(&d_originalVertices, vertexCount * sizeof(float3));
    cudaMemcpy(d_originalVertices, originalVertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);

    int tileSize = 8;
    int tilesInX = camera.imageWidth / tileSize;
    int tilesInY = camera.imageHeight / tileSize;
    int totalTiles = tilesInX * tilesInY;
    
    int *d_tileRangeStart = nullptr, *d_tileRangeEnd = nullptr;
    cudaMalloc(&d_tileRangeStart, totalTiles * sizeof(int));
    cudaMalloc(&d_tileRangeEnd, totalTiles * sizeof(int));

    MouseState mouseState;
    bool running = true;
    SDL_Event event;

    FPSCounter fpsCounter;


    // Copy updated data to GPU
    cudaMemcpy(d_splats, h_splats.data(), vertexCount * sizeof(Gaussian3D), cudaMemcpyHostToDevice);

    while (running) {
        app.processEvents(mouseState, running, event);

        // We can set d_inVertices and d_outVertices to the same pointer because each thread 
        // maps to a different vertex in the array
        rotateVerticesOnGPU(
            d_originalVertices, d_vertices, vertexCount, d_splats,
            mouseState.totalRotationX, mouseState.totalRotationY, boundingSphere.center
        );

        // TODO: logically veryfiy we need to sync
        cudaDeviceSynchronize();
        cudaMemset(d_image, 0, camera.imageWidth * camera.imageHeight * sizeof(float4));  

        // Project Gaussians
        int blockSize = tileSize*tileSize;
        int gridSize = (vertexCount + blockSize - 1) / blockSize;
        projectGaussiansKernel<<<gridSize, blockSize>>>(
            d_splats, d_outSplats, vertexCount, camera, tileSize
        );
        cudaDeviceSynchronize();

        // Sort by tileID then depth
        thrust::device_vector<unsigned long long> d_keys(vertexCount);
        thrust::transform(
            thrust::device_pointer_cast(d_outSplats),
            thrust::device_pointer_cast(d_outSplats + vertexCount),
            d_keys.begin(),
            [] __device__ (const ProjectedSplat& s) {
                return packTileDepth(s.tileID, s.depth);
            }
        );

        thrust::device_ptr<ProjectedSplat> d_splats_ptr(d_outSplats);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_splats_ptr);


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
            [] __device__ (const ProjectedSplat &s) {
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

        // Render
        dim3 blocks(totalTiles, 1, 1);
        dim3 threads(tileSize * tileSize, 1, 1);
        tiledBlendingKernel<<<blocks, threads>>>(
            d_outSplats, d_image, d_tileRangeStart, d_tileRangeEnd,
            camera, tileSize
        );
        cudaDeviceSynchronize();

        // Copy result back and update texture
        std::vector<float4> h_image(camera.imageWidth * camera.imageHeight);
        cudaMemcpy(h_image.data(), d_image, camera.imageWidth * camera.imageHeight * sizeof(float4), cudaMemcpyDeviceToHost);

        std::vector<Uint32> pixels(camera.imageWidth * camera.imageHeight);
        for (int i = 0; i < camera.imageWidth * camera.imageHeight; i++) {
            float4 px = h_image[i];
            Uint8 r = (Uint8)(255.f * fminf(fmaxf(px.x, 0.f), 1.f));
            Uint8 g = (Uint8)(255.f * fminf(fmaxf(px.y, 0.f), 1.f));
            Uint8 b = (Uint8)(255.f * fminf(fmaxf(px.z, 0.f), 1.f));
            Uint8 a = (Uint8)(255.f * fminf(fmaxf(px.w, 0.f), 1.f));
            //Creates RGBA ordering in memory, but SDL interprets this as ABGR (reading right-to-left)
            //pixels[i] = (r << 24) | (g << 16) | (b << 8) | a;
            //- Creates ABGR ordering in memory, SDL correctly interprets as RGBA
            pixels[i] = (r << 0) | (g << 8) | (b << 16) | (a << 24); 
        }

        app.renderFrame(pixels, fpsCounter, camera);
    }

    // Cleanup
    cudaFree(d_splats);
    cudaFree(d_outSplats);
    cudaFree(d_vertices);
    cudaFree(d_image);
    cudaFree(d_tileRangeStart);
    cudaFree(d_tileRangeEnd);
    cudaFree(d_originalVertices);

    return 0;
}