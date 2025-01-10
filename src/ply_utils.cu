#include <iostream>
#include <fstream>
#include <cstring>

#include "gaussian.hpp"
#include "tinyply.h"

int read_init_ply(std::vector<float3>& originalVertices, 
    int &vertexCount, std::vector<float3>& originalColors,
    std::vector<Gaussian3D>& h_splats) {
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

    h_splats.resize(vertexCount);

    for (int i = 0; i < vertexCount; i++) {
        h_splats[i].position = originalVertices[i];
        h_splats[i].scale = make_float3(0.9f, 0.9f, 0.9f);
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
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}