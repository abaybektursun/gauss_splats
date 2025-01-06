#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#include "camera.hpp"
#include "gaussian.hpp"
#include "render_utils.hpp"
#include "cuda_helpers.hpp"

// Forward-declare your PLY loading function
bool loadPlyFile(const std::string& filePath,
                 std::vector<float3>& outVertices,
                 std::vector<int>& outFaces,
                 int& vertexCount, int& faceCount);


struct BoundingSphere {
    float3 center;
    float radius;
};

struct MouseState {
    bool leftButtonDown = false;
    int lastX = 0;
    int lastY = 0;
    float rotationX = 0.0f;
    float rotationY = 0.0f;
};

BoundingSphere calculateBoundingSphere(const std::vector<float3>& vertices) {
    BoundingSphere sphere;
    
    // Calculate center as average of all vertices
    sphere.center = make_float3(0.0f, 0.0f, 0.0f);
    for (const auto& v : vertices) {
        sphere.center.x += v.x;
        sphere.center.y += v.y;
        sphere.center.z += v.z;
    }
    sphere.center.x /= vertices.size();
    sphere.center.y /= vertices.size();
    sphere.center.z /= vertices.size();
    
    // Calculate radius as maximum distance from center to any vertex
    sphere.radius = 0.0f;
    for (const auto& v : vertices) {
        float dx = v.x - sphere.center.x;
        float dy = v.y - sphere.center.y;
        float dz = v.z - sphere.center.z;
        float dist = sqrt(dx*dx + dy*dy + dz*dz);
        sphere.radius = std::max(sphere.radius, dist);
    }
    
    return sphere;
}

void handleMouseEvent(SDL_Event& event, MouseState& mouseState) {
    switch(event.type) {
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                mouseState.leftButtonDown = true;
                mouseState.lastX = event.button.x;
                mouseState.lastY = event.button.y;
            }
            break;
            
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                mouseState.leftButtonDown = false;
            }
            break;
            
        case SDL_MOUSEMOTION:
            if (mouseState.leftButtonDown) {
                int deltaX = event.motion.x - mouseState.lastX;
                int deltaY = event.motion.y - mouseState.lastY;
                
                mouseState.rotationX += deltaY * 0.005f;
                mouseState.rotationY += deltaX * 0.005f;
                
                mouseState.lastX = event.motion.x;
                mouseState.lastY = event.motion.y;
            }
            break;
    }
}

void rotateVertices(std::vector<float3>& vertices, const MouseState& mouseState, const float3& center) {
    // Create rotation matrix around center point
    glm::mat4 toOrigin = glm::translate(glm::mat4(1.0f), glm::vec3(-center.x, -center.y, -center.z));
    glm::mat4 fromOrigin = glm::translate(glm::mat4(1.0f), glm::vec3(center.x, center.y, center.z));
    
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), mouseState.rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
    rotation = glm::rotate(rotation, mouseState.rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
    
    glm::mat4 transform = fromOrigin * rotation * toOrigin;
    
    for (auto& vertex : vertices) {
        glm::vec4 rotated = transform * glm::vec4(vertex.x, vertex.y, vertex.z, 1.0f);
        vertex.x = rotated.x;
        vertex.y = rotated.y;
        vertex.z = rotated.z;
    }
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Gaussian Splats Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        512, 512,
        SDL_WINDOW_SHOWN
    );

    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        512, 512
    );

    if (!texture) {
        std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Load PLY file
    std::string file_path = "/workspaces/gauss_splats/airplane.ply.txt";
    std::vector<float3> originalVertices;
    std::vector<int> faces;
    int vertexCount = 0, faceCount = 0;
    
    if (!loadPlyFile(file_path, originalVertices, faces, vertexCount, faceCount)) {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Calculate bounding sphere once
    BoundingSphere boundingSphere = calculateBoundingSphere(originalVertices);
    
    // Set up fixed camera parameters based on bounding sphere
    OrthoCameraParams camera;
    camera.imageWidth = 512;
    camera.imageHeight = 512;
    
    // Calculate fixed camera bounds that will work for any rotation
    float aspectRatio = float(camera.imageWidth) / float(camera.imageHeight);
    float margin = 1.1f; // Add some margin around the object
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
    
    cudaMalloc(&d_image, 512 * 512 * sizeof(float4));
    cudaMalloc(&d_splats, vertexCount * sizeof(Gaussian3D));
    cudaMalloc(&d_outSplats, vertexCount * sizeof(ProjectedSplat));
    cudaMalloc(&d_vertices, vertexCount * sizeof(float3));

    int tileSize = 16;
    int tilesInX = 512 / tileSize;
    int tilesInY = 512 / tileSize;
    int totalTiles = tilesInX * tilesInY;
    
    int *d_tileRangeStart = nullptr, *d_tileRangeEnd = nullptr;
    cudaMalloc(&d_tileRangeStart, totalTiles * sizeof(int));
    cudaMalloc(&d_tileRangeEnd, totalTiles * sizeof(int));

    MouseState mouseState;
    bool running = true;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            handleMouseEvent(event, mouseState);
        }

        // Create a copy of vertices for this frame and rotate them
        std::vector<float3> vertices = originalVertices;
        rotateVertices(vertices, mouseState, boundingSphere.center);

        // Update Gaussians with rotated positions
        std::vector<Gaussian3D> h_splats(vertexCount);
        for (int i = 0; i < vertexCount; i++) {
            h_splats[i].position = vertices[i];
            h_splats[i].scale = make_float3(0.05f, 0.05f, 0.05f);
            h_splats[i].opacity = 0.5f;
            h_splats[i].color = make_float3(
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX
            );
            h_splats[i].intensity = 0.8f;
        }

        // Clear image and copy updated data to GPU
        cudaMemset(d_image, 0, 512 * 512 * sizeof(float4));
        cudaMemcpy(d_splats, h_splats.data(), vertexCount * sizeof(Gaussian3D), cudaMemcpyHostToDevice);

        // Project Gaussians
        int blockSize = 256;
        int gridSize = (vertexCount + blockSize - 1) / blockSize;
        projectGaussiansKernel<<<gridSize, blockSize>>>(
            d_splats, d_outSplats, vertexCount, camera, 16
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

        // Compute and copy tile ranges
        std::vector<ProjectedSplat> h_splatsSorted(vertexCount);
        cudaMemcpy(h_splatsSorted.data(), d_outSplats, vertexCount * sizeof(ProjectedSplat), cudaMemcpyDeviceToHost);

        std::vector<int> h_tileRangeStart(totalTiles, -1);
        std::vector<int> h_tileRangeEnd(totalTiles, -1);
        computeTileRanges(h_splatsSorted, totalTiles, h_tileRangeStart, h_tileRangeEnd);

        cudaMemcpy(d_tileRangeStart, h_tileRangeStart.data(), totalTiles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tileRangeEnd, h_tileRangeEnd.data(), totalTiles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outSplats, h_splatsSorted.data(), vertexCount * sizeof(ProjectedSplat), cudaMemcpyHostToDevice);

        // Render
        dim3 blocks(totalTiles, 1, 1);
        dim3 threads(tileSize * tileSize, 1, 1);
        tiledBlendingKernel<<<blocks, threads>>>(
            d_outSplats, d_image, d_tileRangeStart, d_tileRangeEnd,
            camera, tileSize
        );
        cudaDeviceSynchronize();

        // Copy result back and update texture
        std::vector<float4> h_image(512 * 512);
        cudaMemcpy(h_image.data(), d_image, 512 * 512 * sizeof(float4), cudaMemcpyDeviceToHost);

        std::vector<Uint32> pixels(512 * 512);
        for (int i = 0; i < 512 * 512; i++) {
            float4 px = h_image[i];
            Uint8 r = (Uint8)(255.f * fminf(fmaxf(px.x, 0.f), 1.f));
            Uint8 g = (Uint8)(255.f * fminf(fmaxf(px.y, 0.f), 1.f));
            Uint8 b = (Uint8)(255.f * fminf(fmaxf(px.z, 0.f), 1.f));
            Uint8 a = (Uint8)(255.f * fminf(fmaxf(px.w, 0.f), 1.f));
            pixels[i] = (r << 24) | (g << 16) | (b << 8) | a;
        }

        SDL_UpdateTexture(texture, NULL, pixels.data(), 512 * sizeof(Uint32));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        SDL_Delay(16); // roughly 60 FPS
    }

    // Cleanup
    cudaFree(d_splats);
    cudaFree(d_outSplats);
    cudaFree(d_vertices);
    cudaFree(d_image);
    cudaFree(d_tileRangeStart);
    cudaFree(d_tileRangeEnd);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}