#include <iostream>
#include <random>
#include <vector>
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

int main()
{
    // Example usage: load a PLY file
    std::string file_path = "/workspaces/gauss_splats/airplane.ply.txt";

    std::vector<float3> vertices;
    std::vector<int> faces;
    int vertexCount = 0, faceCount = 0;
    if (!loadPlyFile(file_path, vertices, faces, vertexCount, faceCount)) {
        return 1;
    }

    std::cout << "Vertices: " << vertexCount 
              << ", Faces: " << faceCount << "\n";

    // Possibly do some CPU debug prints on camera
    OrthoCameraParams camera;
    camera.xMin = 0.f;    // We will compute these properly below
    camera.xMax = 0.f;
    camera.yMin = 0.f;
    camera.yMax = 0.f;
    camera.imageWidth  = 512;
    camera.imageHeight = 512;

    // Move vertices to device to find bounding box
    float3* d_vertices = nullptr;
    cudaMalloc(&d_vertices, vertexCount * sizeof(float3));
    cudaMemcpy(d_vertices, vertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);

    // Use thrust to find bounding box
    thrust::device_vector<float3> d_vec(d_vertices, d_vertices + vertexCount);  // or transform
    thrust::device_vector<float> d_x(vertexCount), d_y(vertexCount);

    thrust::transform(d_vec.begin(), d_vec.end(), d_x.begin(), [] __device__ (float3 v) { return v.x; });
    thrust::transform(d_vec.begin(), d_vec.end(), d_y.begin(), [] __device__ (float3 v) { return v.y; });

    float margin = 0.1f;
    float x_min = *thrust::min_element(d_x.begin(), d_x.end()) - margin;
    float x_max = *thrust::max_element(d_x.begin(), d_x.end()) + margin;
    float y_min = *thrust::min_element(d_y.begin(), d_y.end()) - margin;
    float y_max = *thrust::max_element(d_y.begin(), d_y.end()) + margin;

    camera.xMin = x_min;
    camera.xMax = x_max;
    camera.yMin = y_min;
    camera.yMax = y_max;

    printCameraParams(camera);

    // Build an array of Gaussians (one per vertex, for example)
    std::vector<Gaussian3D> h_splats(vertexCount);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> scale_dist(0.1f, 0.5f);
    std::uniform_real_distribution<float> opacity_dist(0.1f, 1.0f);

    for (int i = 0; i < vertexCount; i++) {
        h_splats[i].position = vertices[i];
        h_splats[i].scale    = make_float3(scale_dist(rng), scale_dist(rng), scale_dist(rng));
        h_splats[i].opacity  = opacity_dist(rng);
        h_splats[i].color    = make_float3(1.0f, 0.0f, 0.0f); // Red
        h_splats[i].intensity= opacity_dist(rng);
    }

    // Copy Gaussians to device
    Gaussian3D* d_splats = nullptr;
    cudaMalloc(&d_splats, vertexCount * sizeof(Gaussian3D));
    cudaMemcpy(d_splats, h_splats.data(), vertexCount*sizeof(Gaussian3D), cudaMemcpyHostToDevice);

    // Prepare output image buffer
    int width  = camera.imageWidth;
    int height = camera.imageHeight;
    std::vector<float4> h_image(width*height, make_float4(0.f, 0.f, 0.f, 1.f));  // RGBA

    float4* d_image = nullptr;
    cudaMalloc(&d_image, width*height*sizeof(float4));
    cudaMemcpy(d_image, h_image.data(), width*height*sizeof(float4), cudaMemcpyHostToDevice);

    // Project Gaussians
    ProjectedSplat* d_outSplats = nullptr;
    cudaMalloc(&d_outSplats, vertexCount*sizeof(ProjectedSplat));

    int blockSize = 256;
    int gridSize  = (vertexCount + blockSize - 1) / blockSize;
    projectGaussiansKernel<<<gridSize, blockSize>>>(
        d_splats, d_outSplats, vertexCount, camera, 16
    );
    cudaDeviceSynchronize();

    // Sort by tileID then by depth (as 64-bit key)
    thrust::device_vector<unsigned long long> d_keys(vertexCount);
    thrust::transform(
        thrust::device_pointer_cast(d_outSplats),
        thrust::device_pointer_cast(d_outSplats + vertexCount),
        d_keys.begin(),
        [] __device__ (const ProjectedSplat& s) {
            // pack tileID (high bits) + approximate depth (low bits)
            // this is simplistic. Consider better ways for floats -> bits
            unsigned long long tilePart = (static_cast<unsigned long long>(s.tileID) & 0xFFFFFFFFull) << 32;
            unsigned long long depthPart= static_cast<unsigned long long>(s.depth * 1e6) & 0xFFFFFFFFull;
            return (tilePart | depthPart);
        }
    );

    thrust::device_ptr<ProjectedSplat> d_splats_ptr(d_outSplats);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_splats_ptr);

    // Copy sorted splats back to host
    std::vector<ProjectedSplat> h_splatsSorted(vertexCount);
    cudaMemcpy(h_splatsSorted.data(), d_outSplats,
               vertexCount*sizeof(ProjectedSplat),
               cudaMemcpyDeviceToHost);

    // Compute tile range
    int tileSize   = 16;
    int tilesInX   = width / tileSize;
    int tilesInY   = height / tileSize;
    int totalTiles = tilesInX * tilesInY;

    std::vector<int> h_tileRangeStart(totalTiles, -1);
    std::vector<int> h_tileRangeEnd  (totalTiles, -1);

    computeTileRanges(h_splatsSorted, totalTiles,
                      h_tileRangeStart, h_tileRangeEnd);

    // Copy tile ranges to device
    int *d_tileRangeStart = nullptr, *d_tileRangeEnd = nullptr;
    cudaMalloc(&d_tileRangeStart, totalTiles*sizeof(int));
    cudaMalloc(&d_tileRangeEnd,   totalTiles*sizeof(int));
    cudaMemcpy(d_tileRangeStart, h_tileRangeStart.data(),
               totalTiles*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tileRangeEnd,   h_tileRangeEnd.data(),
               totalTiles*sizeof(int), cudaMemcpyHostToDevice);

    // Copy sorted splats array to device
    cudaMemcpy(d_outSplats, h_splatsSorted.data(),
               vertexCount*sizeof(ProjectedSplat),
               cudaMemcpyHostToDevice);

    // Tiled blending kernel
    dim3 blocks(totalTiles, 1, 1);
    dim3 threads(tileSize*tileSize, 1, 1);
    tiledBlendingKernel<<<blocks, threads>>>(
        d_outSplats, d_image, d_tileRangeStart, d_tileRangeEnd,
        camera, tileSize
    );
    cudaDeviceSynchronize();

    // Copy final image back
    cudaMemcpy(h_image.data(), d_image,
               width*height*sizeof(float4),
               cudaMemcpyDeviceToHost);

    // [Optional] Save or display the final h_image.

    // Cleanup
    cudaFree(d_splats);
    cudaFree(d_outSplats);
    cudaFree(d_vertices);
    cudaFree(d_image);
    cudaFree(d_tileRangeStart);
    cudaFree(d_tileRangeEnd);

    std::cout << "Done.\n";
    return 0;
}
