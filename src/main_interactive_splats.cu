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
#include <thrust/iterator/discard_iterator.h>

#include "camera.hpp"
#include "gaussian.hpp"
#include "render_utils.hpp"
#include "cuda_helpers.hpp"
#include "camera_gpu.cu"

#include "tinyply.h"

const int WINDOW_WIDTH = 512*1.5;
const int WINDOW_HEIGHT = 512*1.5;

struct FPSCounter {
    Uint32 frameCount = 0;
    Uint32 lastTime = 0;
    float currentFPS = 0.0f;
    
    void update() {
        frameCount++;
        Uint32 currentTime = SDL_GetTicks();
        
        // Update FPS every second
        if (currentTime - lastTime > 1000) {
            currentFPS = frameCount * 1000.0f / (currentTime - lastTime);
            printf("\rFPS: %.1f", currentFPS);
            fflush(stdout);
            
            frameCount = 0;
            lastTime = currentTime;
        }
    }
};


struct BoundingSphere {
    float3 center;
    float radius;
};

struct MouseState {
    bool leftButtonDown = false;
    int lastX = 0;
    int lastY = 0;
    float totalRotationX = 0.0f;  // Track total rotation
    float totalRotationY = 0.0f;
    float lastRotationX = 0.0f;   // Track last frame's rotation
    float lastRotationY = 0.0f;
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
                // Store current rotation when starting drag
                mouseState.lastRotationX = mouseState.totalRotationX;
                mouseState.lastRotationY = mouseState.totalRotationY;
            }
            break;
            
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                mouseState.leftButtonDown = false;
            }
            break;
            
        case SDL_MOUSEMOTION:
            if (mouseState.leftButtonDown) {
                // Convert mouse position to rotation angles
                mouseState.totalRotationX = mouseState.lastRotationX + 
                    (event.motion.y - mouseState.lastY) * (2.0f * M_PI / WINDOW_HEIGHT);
                mouseState.totalRotationY = mouseState.lastRotationY + 
                    (event.motion.x - mouseState.lastX) * (2.0f * M_PI / WINDOW_WIDTH);
            }
            break;
    }
}

/*
void rotateVertices(std::vector<float3>& vertices, const MouseState& mouseState, const float3& center) {
    // 1) Translate the object so that 'center' is at the origin (0,0,0).
    //    This lets us rotate around the center.
    glm::mat4 toOrigin = glm::translate(glm::mat4(1.0f),
                                        glm::vec3(-center.x, -center.y, -center.z));

    // 2) Translate back from the origin to the original 'center' location
    //    after the rotation. Essentially an inverse of 'toOrigin'.
    glm::mat4 fromOrigin = glm::translate(glm::mat4(1.0f),
                                          glm::vec3(center.x, center.y, center.z));

    // 3) Create a rotation matrix around the Y-axis by 'mouseState.rotationY'.
    //    GLM angle param is in radians, so if rotationY is e.g. 0.5, that’s ~28.65°.
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f),
                                     mouseState.rotationY,
                                     glm::vec3(0.0f, 1.0f, 0.0f));

    // 4) Then rotate around the X-axis by 'mouseState.rotationX'.
    //    This modifies the same matrix 'rotation'.
    rotation = glm::rotate(rotation,
                           mouseState.rotationX,
                           glm::vec3(1.0f, 0.0f, 0.0f));

    // 5) Combine all transforms:
    //    - Move the model to origin
    //    - Rotate
    //    - Move back
    //    The final transform matrix is 'fromOrigin * rotation * toOrigin'.
    glm::mat4 transform = fromOrigin * rotation * toOrigin;

    // 6) Apply this transform to every vertex in the vector.
    //    Each vertex is treated as (x, y, z, 1) in homogeneous coordinates,
    //    then multiplied by the 4x4 matrix.
    for (auto& vertex : vertices) {
        glm::vec4 rotated = transform * glm::vec4(vertex.x, vertex.y, vertex.z, 1.0f);
        vertex.x = rotated.x;
        vertex.y = rotated.y;
        vertex.z = rotated.z;
    }
}
*/

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Gaussian Splats Viewer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
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

    // Replace texture creation with:
    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT
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
    std::vector<float3> originalColors;
    int vertexCount = 0, faceCount = 0;
    
    // OLD CODE
    /*if (!loadPlyFile(file_path, originalVertices, faces, vertexCount, faceCount)) {
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }*/

    // New code 
    try {
        // Open the PLY file
        std::ifstream file("/workspaces/gauss_splats/dragon.ply", std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open PLY file.");

        tinyply::PlyFile plyFile;
        plyFile.parse_header(file);

        // Read vertex properties
        std::shared_ptr<tinyply::PlyData> vertices, colors;
        vertices = plyFile.request_properties_from_element("vertex", {"x", "y", "z"});
        colors = plyFile.request_properties_from_element("vertex", {"red", "green", "blue"});

        plyFile.read(file);

        // Process vertex data
        std::vector<float> vertexBuffer(vertices->count * 3);
        std::memcpy(vertexBuffer.data(), vertices->buffer.get(), vertices->buffer.size_bytes());

        std::vector<uint8_t> colorBuffer(colors->count * 3);
        std::memcpy(colorBuffer.data(), colors->buffer.get(), colors->buffer.size_bytes());

        std::cout << "Read " << vertices->count << " points with color data." << std::endl;

        // Convert and copy data to originalVertices
        originalVertices.resize(vertices->count);
        for (size_t i = 0; i < vertices->count; i++) {
            originalVertices[i] = make_float3(
                vertexBuffer[i * 3 + 0],
                vertexBuffer[i * 3 + 1],
                vertexBuffer[i * 3 + 2]
            );
        }
        vertexCount = vertices->count;

        // Same for colors
        originalColors.resize(vertices->count);
        for (size_t i = 0; i < vertices->count; i++) {
            originalColors[i] = make_float3(
                colorBuffer[i * 3 + 0] / 255.0f,
                colorBuffer[i * 3 + 1] / 255.0f,
                colorBuffer[i * 3 + 2] / 255.0f
            );
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Calculate bounding sphere once
    BoundingSphere boundingSphere = calculateBoundingSphere(originalVertices);
    
    // Set up fixed camera parameters based on bounding sphere
    OrthoCameraParams camera;
    camera.imageWidth = WINDOW_WIDTH;
    camera.imageHeight = WINDOW_HEIGHT;
    
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

    std::vector<Gaussian3D> h_splats(vertexCount);
    for (int i = 0; i < vertexCount; i++) {
        h_splats[i].position = originalVertices[i];
        h_splats[i].scale = make_float3(0.1f, 0.1f, 0.1f);
        h_splats[i].opacity = 1.0f;
        // Read the colors from the original data
        h_splats[i].color = make_float3(
            originalColors[i].x,
            originalColors[i].y,
            originalColors[i].z
        );
        h_splats[i].intensity = 0.0f;
        h_splats[i].rotation = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Copy updated data to GPU
    cudaMemcpy(d_splats, h_splats.data(), vertexCount * sizeof(Gaussian3D), cudaMemcpyHostToDevice);

    // TODO: remove
    int TMP_ITERS = 500;
    while (running) {
        // TODO: remove
        if (TMP_ITERS-- <= 0) {
            break;
        }
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            handleMouseEvent(event, mouseState);
        }

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

        SDL_UpdateTexture(texture, NULL, pixels.data(), camera.imageWidth * sizeof(Uint32));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        fpsCounter.update();

        //SDL_Delay(16); // roughly 60 FPS
    }

    // Cleanup
    cudaFree(d_splats);
    cudaFree(d_outSplats);
    cudaFree(d_vertices);
    cudaFree(d_image);
    cudaFree(d_tileRangeStart);
    cudaFree(d_tileRangeEnd);
    cudaFree(d_originalVertices);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}