#include <iostream>
#include <random>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

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
        return 1;
    }

    // Load PLY file
    std::vector<float3> originalVertices;
    std::vector<float3> originalColors;
    int vertexCount = 0;
    std::vector<Gaussian3D> h_splats(0); // Will be resized in read_init_ply
    read_init_ply(originalVertices, vertexCount, originalColors, h_splats);
    
    BoundingSphere boundingSphere = calculateBoundingSphere(originalVertices);
    OrthoCameraParams camera;
    camera.imageWidth = WINDOW_WIDTH;
    camera.imageHeight = WINDOW_HEIGHT;

    
    // Calculate fixed camera bounds that will work for any rotation
    float aspectRatio = float(camera.imageWidth) / float(camera.imageHeight);
    float margin = 1.0f; // >1.0 to add some margin around the object
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
    GPUData gpuData;

    const int tileSize = 8;
    const int tilesInX = camera.imageWidth / tileSize;
    const int tilesInY = camera.imageHeight / tileSize;
    const int totalTiles = tilesInX * tilesInY;
    const int blockSize = tileSize*tileSize;
    const int gridSize = (vertexCount + blockSize - 1) / blockSize;
    
    cudaMalloc(&gpuData.d_tileRangeStart, totalTiles * sizeof(int));
    cudaMalloc(&gpuData.d_tileRangeEnd, totalTiles * sizeof(int));
    cudaMalloc(&gpuData.d_image, camera.imageWidth * camera.imageHeight * sizeof(float4));
    cudaMalloc(&gpuData.d_outSplats, vertexCount * sizeof(ProjectedSplat));
    cudaMalloc(&gpuData.d_vertices, vertexCount * sizeof(float3));
    cudaMemcpy(gpuData.d_vertices, originalVertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);

    cudaMalloc(&gpuData.d_originalVertices, vertexCount * sizeof(float3));
    cudaMemcpy(gpuData.d_originalVertices, originalVertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpuData.d_splats, vertexCount * sizeof(Gaussian3D));
    cudaMemcpy(gpuData.d_splats, h_splats.data(), vertexCount * sizeof(Gaussian3D), cudaMemcpyHostToDevice);


    SDL_Event event;
    bool running = true;
    MouseState mouseState;
    while (running) {
        app.processEvents(mouseState, running, event);

        // We can set d_inVertices and d_outVertices to the same pointer because each thread 
        // maps to a different vertex in the array
        rotateVerticesOnGPU(
            gpuData.d_originalVertices, gpuData.d_vertices, vertexCount, gpuData.d_splats,
            mouseState.totalRotationX, mouseState.totalRotationY, boundingSphere.center
        );

        // Clear image
        cudaMemset(gpuData.d_image, 0, camera.imageWidth * camera.imageHeight * sizeof(float4));  

        // Project Gaussians
        projectGaussiansKernel<<<gridSize, blockSize>>>(
            gpuData.d_splats, gpuData.d_outSplats, vertexCount, camera, tileSize
        );
        cudaDeviceSynchronize();

        // Sort by tileID then depth
        sortSplats(gpuData, vertexCount);

        generateTileRanges(gpuData.d_outSplats, totalTiles, tileSize, vertexCount, gpuData.d_tileRangeStart, gpuData.d_tileRangeEnd);

        // Render
        dim3 blocks(totalTiles, 1, 1);
        dim3 threads(tileSize * tileSize, 1, 1);
        tiledBlendingKernel<<<blocks, threads>>>(
            gpuData.d_outSplats, gpuData.d_image, gpuData.d_tileRangeStart, gpuData.d_tileRangeEnd,
            camera, tileSize
        );
        cudaDeviceSynchronize();

        // Copy result back and update texture
        std::vector<float4> h_image(camera.imageWidth * camera.imageHeight);
        cudaMemcpy(h_image.data(), gpuData.d_image, camera.imageWidth * camera.imageHeight * sizeof(float4), cudaMemcpyDeviceToHost);

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

        app.renderFrame(pixels, camera);
    }

    releaseGPUData(gpuData);

    return 0;
}