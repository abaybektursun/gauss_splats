#include <iostream>
#include <fstream>
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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
    thrust::device_ptr<float3> dev_ptr(d_vertices);
    thrust::device_vector<float3> d_vec(dev_ptr, dev_ptr + vertexCount);

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

    // DEBUG: Print CPU bounding box................................
    float cpuXmin, cpuXmax, cpuYmin, cpuYmax;
    cpuXmin = cpuXmax = vertices[0].x;
    cpuYmin = cpuYmax = vertices[0].y;
    for (int i = 1; i < vertexCount; i++) {
        cpuXmin = std::min(cpuXmin, vertices[i].x);
        cpuXmax = std::max(cpuXmax, vertices[i].x);
        cpuYmin = std::min(cpuYmin, vertices[i].y);
        cpuYmax = std::max(cpuYmax, vertices[i].y);
    }
    std::cout << "CPU bbox: [" << cpuXmin << "," << cpuXmax << "]"
            << " x [" << cpuYmin << "," << cpuYmax << "]\n";
    // ...............................................................

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
    measureTime("Copy Gaussians to device", [&] {
        cudaMalloc(&d_splats, vertexCount * sizeof(Gaussian3D));
        cudaMemcpy(d_splats, h_splats.data(), vertexCount*sizeof(Gaussian3D), cudaMemcpyHostToDevice);
    });

    // Prepare output image buffer
    int width  = camera.imageWidth;
    int height = camera.imageHeight;
    std::vector<float4> h_image(width*height, make_float4(0.f, 0.f, 0.f, 0.f));  // RGBA
    float4* d_image = nullptr;
    measureTime("Prepare output image buffer", [&] {
        cudaMalloc(&d_image, width*height*sizeof(float4));
        cudaMemcpy(d_image, h_image.data(), width*height*sizeof(float4), cudaMemcpyHostToDevice);
    });

    // Project Gaussians
    ProjectedSplat* d_outSplats = nullptr;
    measureTime("Project Gaussians", [&] {
        cudaMalloc(&d_outSplats, vertexCount*sizeof(ProjectedSplat));
        int blockSize = 256;
        int gridSize  = (vertexCount + blockSize - 1) / blockSize;
        projectGaussiansKernel<<<gridSize, blockSize>>>(
            d_splats, d_outSplats, vertexCount, camera, 16
        );
        cudaDeviceSynchronize();
    });

    // DEBUG: Save the projected splats to a text file
    std::ofstream ofs_splat("splats.txt");
    std::vector<ProjectedSplat> h_splats_debug(vertexCount);
    cudaMemcpy(h_splats_debug.data(), d_outSplats, vertexCount*sizeof(ProjectedSplat), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vertexCount; i++) {
        ofs_splat << h_splats_debug[i].tileID << " "
            << h_splats_debug[i].depth << " "
            << h_splats_debug[i].pixelX << " "
            << h_splats_debug[i].pixelY << "\n";
    }

    // DEBUG: Count valid splats
    int validCount = 0;
    for (auto& s : h_splats_debug) {
        if (s.tileID >= 0) validCount++;
    }
    std::cout << "Valid splats in view: " << validCount << " / " << vertexCount << "\n";



    // Sort by tileID then by depth (as 64-bit key)
    thrust::device_vector<unsigned long long> d_keys(vertexCount);
    measureTime("Sort splats", [&] {
        // Transform each ProjectedSplat into a 64-bit key for sorting:
        thrust::transform(
            thrust::device_pointer_cast(d_outSplats),
            thrust::device_pointer_cast(d_outSplats + vertexCount),
            d_keys.begin(),
            [] __device__ (const ProjectedSplat& s)
            {
                // tileID in high bits, depth in low bits
                // so that we sort primarily by tileID, and secondarily by depth
                return packTileDepth(s.tileID, s.depth);
            }
        );
    });

    
    // Now sort by the keys:
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
    measureTime("computeTileRanges", [&] {
        computeTileRanges(h_splatsSorted, totalTiles,
                        h_tileRangeStart, h_tileRangeEnd);
    });

    // Copy tile ranges to device
    int *d_tileRangeStart = nullptr, *d_tileRangeEnd = nullptr;
    measureTime("Copy tile ranges to device", [&] {
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
    });

    // Tiled blending kernel
    dim3 blocks(totalTiles, 1, 1);
    dim3 threads(tileSize*tileSize, 1, 1);
    measureTime("Tiled blending kernel", [&] {
        tiledBlendingKernel<<<blocks, threads>>>(
            d_outSplats, d_image, d_tileRangeStart, d_tileRangeEnd,
            camera, tileSize
        );
        cudaDeviceSynchronize();
    });

    // Copy final image back
    cudaMemcpy(h_image.data(), d_image,
               width*height*sizeof(float4),
               cudaMemcpyDeviceToHost);

    // Save image to disk png and jpg
    // Convert float4 [0..1] to 8-bit RGBA
    std::vector<unsigned char> outImage(width*height*4);
    for(int i = 0; i < width*height; i++) {
        float4 px = h_image[i];
        outImage[4*i + 0] = (unsigned char)(255.f * fminf(fmaxf(px.x, 0.f), 1.f));
        outImage[4*i + 1] = (unsigned char)(255.f * fminf(fmaxf(px.y, 0.f), 1.f));
        outImage[4*i + 2] = (unsigned char)(255.f * fminf(fmaxf(px.z, 0.f), 1.f));
        outImage[4*i + 3] = (unsigned char)(255.f * fminf(fmaxf(px.w, 0.f), 1.f));
    }
    stbi_write_png("output.png", width, height, 4, outImage.data(), width*4);
    stbi_write_jpg("output.jpg", width, height, 4, outImage.data(), 100);


    // Resize the image matrx to 32x32, simple on CPU
    int newWidth = 32;
    int newHeight = 32;
    std::vector<float4> h_imageResized(newWidth*newHeight, make_float4(0.f, 0.f, 0.f, 0.f));
    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            int oldX = x * width / newWidth;
            int oldY = y * height / newHeight;
            h_imageResized[y*newWidth + x] = h_image[oldY*width + oldX];
        }
    }
    // save the resized image as a text file
    std::ofstream ofs("output32x32.txt");
    for (int i = 0; i < newWidth*newHeight; i++) {
        ofs << h_imageResized[i].x << " "
            << h_imageResized[i].y << " "
            << h_imageResized[i].z << " "
            << h_imageResized[i].w << "\n";
    }



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
