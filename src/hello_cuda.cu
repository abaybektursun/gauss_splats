#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <chrono>
#include <functional>
#include <random>


__global__ void helloKernel() {
    printf("Hello from block %d, thread %d\n",
           blockIdx.x, threadIdx.x);
}



// Wrap This in a Function
__device__ void orthographicProject(float x, float y,
                                    const OrthoCameraParams& cam,
                                    int& outU, int& outV)
{
    float normalizedX = (x - cam.xMin) / (cam.xMax - cam.xMin);
    float normalizedY = (y - cam.yMin) / (cam.yMax - cam.yMin);

    // multiply by image size
    outU = static_cast<int>(normalizedX * (cam.imageWidth - 1));
    outV = static_cast<int>(normalizedY * (cam.imageHeight - 1));
}

// -------------------------
// 2) The alpha blend helper
// -------------------------

// -------------------------
// 3) The tiled blending kernel
// -------------------------
// A small CPU function to compute d_tileRangeStart & d_tileRangeEnd
// from the sorted splats array. This is a host side approach for clarity:



int main() {
    if (false){
        // Launch kernel: 2 blocks, 4 threads each (example)
        
        helloKernel<<<2, 4>>>();
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Just a host print
        std::cout << "Hello from the CPU!\n";

        // Sorting with Thrust -------------------
    
        thrust::host_vector<int> h_vec;
        thrust::device_vector<int> d_vec;
        int num_elements = 1e7;
        std::cout << "Sorting " << num_elements << " elements\n";
        measureTime("Thrust Setup", [&] {
            h_vec.resize(num_elements);
            thrust::generate(h_vec.begin(), h_vec.end(), rand);
        });
        measureTime("Copy to device", [&] {
            d_vec = h_vec; // Copy to device
        });

        measureTime("Thrust Sort", [&] {
            thrust::sort(d_vec.begin(), d_vec.end());
        });

        measureTime("Check sorted", [&] {
            if (thrust::is_sorted(d_vec.begin(), d_vec.end())) {
                std::cout << "Vector is sorted\n";
            } else {
                std::cout << "Vector is not sorted\n";
            }
        });
    }

    // File path
    std::string file_path = "/workspaces/gauss_splats/airplane.ply.txt";
    std::ifstream file(file_path);
    
    if (!file) {
        std::cerr << "Failed to open " << file_path << "\n";
        return 1;
    }

    int vertexCount = 0, faceCount = 0;
    std::string line;
    
    // Read header lines
    while (std::getline(file, line)) {
        std::cout << "[HEADER] " << line << "\n";

        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> vertexCount; // e.g. "element vertex 1234"
        } 
        else if (line.find("element face") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> faceCount;   // e.g. "element face 567"
        } 
        else if (line.find("end_header") != std::string::npos) {
            // Stop reading the header
            break;
        }
    }

    // Read vertices as float3
    std::vector<float3> vertices;
    vertices.reserve(vertexCount);

    for (int i = 0; i < vertexCount; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Failed to read line for vertex " << i 
                    << " (possibly not enough lines in the file?)\n";
            return 1;
        }

        float x, y, z;
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Failed to parse x,y,z from line " 
                    << i << " => \"" << line << "\"\n";
            return 1;
        }

        // Store in float3
        vertices.push_back(make_float3(x, y, z));
    }

    std::cout << "Successfully read all vertices.\n";

    // Read the number of vertices per face
    // (We do NOT consume an extra std::getline here)
    int num_vertices_per_face = 0;
    file >> num_vertices_per_face;
    std::cout << "Num vertices per face: " << num_vertices_per_face << "\n";

    // Prepare to read face indices
    std::vector<int> faces;
    faces.reserve(faceCount * num_vertices_per_face);

    // Now read each face line
    for (int i = 0; i < faceCount; i++) {
        // If the file extraction reached the end, fall back to getline:
        if (!std::getline(file, line)) {
            std::cerr << "Failed to read line for face " << i 
                    << " (possibly not enough lines in the file?)\n";
            return 1;
        }
        // Parse the line
        std::istringstream iss(line);

        // Discard the first number, which is usually "3" or "4" 
        // indicating how many vertices form this face
        int tmp;
        iss >> tmp;

        // Read the indices that follow
        int index;
        while (iss >> index) {
            faces.push_back(index);
        }
    }

    // Verify
    std::cout << "Vertices: " << vertexCount 
            << ", Faces: " << faceCount << "\n";
    std::cout << "First vertex: " 
            << vertices[0].x << ", " 
            << vertices[0].y << ", " 
            << vertices[0].z << "\n";
    std::cout << "Last vertex: " 
            << vertices.back().x << ", " 
            << vertices.back().y << ", " 
            << vertices.back().z << "\n";

    // Print first face
    for (int i = 0; i < num_vertices_per_face; i++) {
        std::cout << faces[i] << " ";
    }
    std::cout << "\n";
    // Print last face
    for (int i = 0; i < num_vertices_per_face; i++) {
        std::cout << faces[faces.size() - num_vertices_per_face + i] << " ";
    }
    std::cout << "\n";

    if (false){
        // Move vertices and faces to the device
        // Each float3 is 12 bytes (3 floats)
        size_t vertex_buffer_size = vertices.size() * sizeof(float3);
        size_t face_buffer_size   = faces.size()   * sizeof(int);

        float3* d_vertices = nullptr;
        int*    d_faces    = nullptr;

        cudaMalloc(&d_vertices, vertex_buffer_size);
        cudaMalloc(&d_faces,    face_buffer_size);

        cudaMemcpy(d_vertices, vertices.data(), vertex_buffer_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_faces,    faces.data(),    face_buffer_size,   cudaMemcpyHostToDevice);

        // Launch kernel to sum vertices
        float result[3] = {0.0f, 0.0f, 0.0f};
        float* d_result = nullptr;
        cudaMalloc(&d_result, 3 * sizeof(float));   
        cudaMemcpy(d_result, result, 3 * sizeof(float), cudaMemcpyHostToDevice);

        sumVertices<<<32, 256>>>(d_vertices, d_result, vertexCount);

        cudaMemcpy(result, d_result, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Sum of vertices: " << result[0] << ", " << result[1] << ", " << result[2] << "\n";

        // Free device memory
        cudaFree(d_vertices);
        cudaFree(d_faces);
        cudaFree(d_result);
    }

    // Fill in splats with the vertices 
    int num_splats = vertexCount;
    // Host-side storage
    std::vector<Gaussian3D> h_splats(num_splats);

    // Fill in splats, one per vertex, with random scale and opacity
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> scale_dist(0.1f, 0.5f);
    std::uniform_real_distribution<float> opacity_dist(0.1f, 1.0f);

    for (int i = 0; i < num_splats; i++) {
        h_splats[i].position = vertices[i];
        h_splats[i].scale = make_float3(scale_dist(rng), scale_dist(rng), scale_dist(rng));
        h_splats[i].opacity = opacity_dist(rng);
        h_splats[i].color = make_float3(1.0f, 0.0f, 0.0f); // Red
        h_splats[i].intensity = opacity_dist(rng);
    }
    

    // Move splats to the device
    size_t splat_buffer_size = num_splats * sizeof(Gaussian3D);
    Gaussian3D* d_splats = nullptr;
    cudaMalloc(&d_splats, splat_buffer_size);
    cudaMemcpy(d_splats, h_splats.data(), splat_buffer_size, cudaMemcpyHostToDevice);


    // Create output image buffer
    int width = 512, height = 512;
    int num_pixels = width * height;
    std::cout << "Creating " << width << "x" << height << " image buffer\n";

    // Host-side storage
    std::vector<float4> h_image(num_pixels);

    // Initialize image buffer to a black background
    for (int i = 0; i < num_pixels; i++) {
        h_image[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Move image buffer to the device
    size_t image_buffer_size = num_pixels * sizeof(float4);
    float4* d_image = nullptr;
    cudaMalloc(&d_image, image_buffer_size);
    cudaMemcpy(d_image, h_image.data(), image_buffer_size, cudaMemcpyHostToDevice);


    // Compute a simlpe Camera/View transform: $\mathrm{projection}(x, y, z) = (x, y)$
    //Keep it very simple for now. We just want to get a point’s (x,y) to a pixel (u,v).
    float3* d_vertices = nullptr;
    cudaMalloc(&d_vertices, vertexCount * sizeof(float3));
    cudaMemcpy(d_vertices, vertices.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice);

    float margin = 0.1f;
    // Find min and maxes using thrust::min_element and thrust::max_element for x, y (on device)
    thrust::device_vector<float> d_x(vertexCount), d_y(vertexCount);
    thrust::transform(d_vertices, d_vertices + vertexCount, d_x.begin(), [] __device__ (float3 v) { return v.x; });
    thrust::transform(d_vertices, d_vertices + vertexCount, d_y.begin(), [] __device__ (float3 v) { return v.y; });
    auto x_min = *thrust::min_element(d_x.begin(), d_x.end());
    auto x_max = *thrust::max_element(d_x.begin(), d_x.end());
    auto y_min = *thrust::min_element(d_y.begin(), d_y.end());
    auto y_max = *thrust::max_element(d_y.begin(), d_y.end());
    // add the margin to the min and max
    x_min -= margin;    x_max += margin;
    y_min -= margin;    y_max += margin;
    std::cout << "x_min: " << x_min << ", x_max: " << x_max << ", y_min: " << y_min << ", y_max: " << y_max << "\n";

    OrthoCameraParams camera;
    camera.xMin = x_min;
    camera.xMax = x_max;
    camera.yMin = y_min;
    camera.yMax = y_max;
    camera.imageWidth = width;
    camera.imageHeight = height;


    // Launch kernel to project splats
    ProjectedSplat* d_outSplats = nullptr;
    cudaMalloc(&d_outSplats, num_splats * sizeof(ProjectedSplat));

    projectGaussiansKernel<<<(num_splats + 255) / 256, 256>>>(d_splats, d_outSplats, num_splats, camera);
    
    // Create a key-value pair for sorting
    thrust::device_vector<uint64_t> d_keys(num_splats); // 64-bit sorting keys

    // Generate keys
    thrust::transform(d_outSplats, d_outSplats + num_splats, d_keys.begin(),
        [] __device__ (ProjectedSplat s) {
            uint64_t key = (static_cast<uint64_t>(s.tileID) << 32) | (static_cast<uint32_t>(s.depth * 1e6));
            return key;
        });
    
    // Sort both the keys & the splats together
    // to get ascending tileID, then ascending depth
    thrust::device_ptr<ProjectedSplat> d_splats_ptr(d_outSplats);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_splats_ptr);

    // Compute tileRangeStart / tileRangeEnd on CPU
    // We'll copy d_outSplats back to host
    std::vector<ProjectedSplat> h_splatsSorted(num_splats);
    cudaMemcpy(h_splatsSorted.data(), d_outSplats, 
               num_splats * sizeof(ProjectedSplat),
               cudaMemcpyDeviceToHost);

    // total tiles:
    int tile_size = 16;
    int tilesInX = width / tile_size;
    int tilesInY = height / tile_size;
    int totalTiles = tilesInX * tilesInY;

    // Create arrays for tileRangeStart & tileRangeEnd
    std::vector<int> h_tileRangeStart(totalTiles, -1);
    std::vector<int> h_tileRangeEnd  (totalTiles, -1);

    computeTileRanges(h_splatsSorted, totalTiles, 
                      h_tileRangeStart, h_tileRangeEnd);

    // Copy these to device
    int *d_tileRangeStart = nullptr, *d_tileRangeEnd = nullptr;
    cudaMalloc(&d_tileRangeStart, totalTiles * sizeof(int));
    cudaMalloc(&d_tileRangeEnd,   totalTiles * sizeof(int));

    cudaMemcpy(d_tileRangeStart, h_tileRangeStart.data(),
               totalTiles*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tileRangeEnd,   h_tileRangeEnd.data(),
               totalTiles*sizeof(int), cudaMemcpyHostToDevice);

    // Now launch tiledBlendingKernel
    // We'll have "totalTiles" blocks, each block is for 1 tile
    // We'll have tile_size * tile_size threads in 1D or 2D.
    dim3 blocks(totalTiles, 1, 1);
    dim3 threads(tile_size * tile_size, 1, 1);  // 1D block

    tiledBlendingKernel<<<blocks, threads>>>( 
        d_outSplats,
        d_image,
        d_tileRangeStart,
        d_tileRangeEnd,
        camera,
        tile_size
    );
    cudaDeviceSynchronize();

    // 4) Copy final image back to CPU
    std::vector<float4> h_finalImage(num_pixels);
    cudaMemcpy(h_finalImage.data(), d_image, 
               num_pixels*sizeof(float4), 
               cudaMemcpyDeviceToHost);
    

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_splats);
    cudaFree(d_vertices);
    cudaFree(d_outSplats);
    
    return 0;
}
