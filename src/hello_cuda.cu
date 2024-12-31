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

void measureTime(const std::string& label, const std::function<void()>& run) {
    auto start = std::chrono::high_resolution_clock::now();
    run();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << label << " took " << ms << " ms\n";
}


__global__ void helloKernel() {
    printf("Hello from block %d, thread %d\n",
           blockIdx.x, threadIdx.x);
}

__global__ void sumVertices(float3* vertices, float* result, int vertexCount) {
    // Thread indexing - fundamental CUDA concept
    // Global thread ID = block index * block size + thread index within block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reduction
    // Each thread in a block will contribute to this shared sum
    __shared__ float3 blockSum;
    
    // Initialize shared memory if this is thread 0 in the block
    if (threadIdx.x == 0) {
        blockSum.x = 0.0f;
        blockSum.y = 0.0f;
        blockSum.z = 0.0f;
    }
    __syncthreads();  // Ensure shared memory is initialized
    
    // Grid-stride loop pattern - handle multiple elements per thread
    for (int i = tid; i < vertexCount; i += gridDim.x * blockDim.x) {
        // Load vertex data
        float3 vertex = vertices[i];
        
        // Accumulate in shared memory
        atomicAdd(&blockSum.x, vertex.x);
        atomicAdd(&blockSum.y, vertex.y);
        atomicAdd(&blockSum.z, vertex.z);
    }
    
    // Wait for all threads in block to finish
    __syncthreads();
    
    // Only thread 0 in each block adds block result to global sum
    if (threadIdx.x == 0) {
        atomicAdd(&result[0], blockSum.x);
        atomicAdd(&result[1], blockSum.y);
        atomicAdd(&result[2], blockSum.z);
    }
}

struct Gaussian3D {
    float3 position;     // (x, y, z)
    float3 scale;        // scale for x, y, z
    float4 rotation;     // quaternion
    float  opacity;      // alpha
    // Later: spherical harmonics for color
};

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

    if (false){
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
    }

    // Generate random Gaussian splats
    if (true){
        int num_splats = 1e6;
        std::cout << "Generating " << num_splats << " random Gaussian splats\n";

        // Host-side storage
        std::vector<Gaussian3D> h_splats(num_splats);

        // Random number generator
        std::default_random_engine rng;
        std::normal_distribution<float> normal(0.0f, 1.0f);

        // Fill in splats
        for (int i = 0; i < num_splats; i++) {
            Gaussian3D& splat = h_splats[i];
            splat.position.x = normal(rng);
            splat.position.y = normal(rng);
            splat.position.z = normal(rng);
            splat.scale.x = 0.1f + 0.1f * normal(rng);
            splat.scale.y = 0.1f + 0.1f * normal(rng);
            splat.scale.z = 0.1f + 0.1f * normal(rng);
            splat.rotation.x = normal(rng);
            splat.rotation.y = normal(rng);
            splat.rotation.z = normal(rng);
            splat.rotation.w = normal(rng);
            splat.opacity = 0.1f + 0.1f * normal(rng);
        }

        // Move splats to the device
        size_t splat_buffer_size = num_splats * sizeof(Gaussian3D);
        Gaussian3D* d_splats = nullptr;
        cudaMalloc(&d_splats, splat_buffer_size);
        cudaMemcpy(d_splats, h_splats.data(), splat_buffer_size, cudaMemcpyHostToDevice);

        // Free device memory
        cudaFree(d_splats);
    }

    return 0;
}
