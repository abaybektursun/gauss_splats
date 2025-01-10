#include <iostream>
#include <fstream>
#include <cstring>
#include <random>

#include "gaussian.hpp"
#include "tinyply.h"

int read_init_ply(std::vector<float3>& originalVertices, 
    int &vertexCount, std::vector<float3>& originalColors,
    std::vector<Gaussian3D>& h_splats, float drop = 0.0f) {
    
    if (drop < 0.0f || drop > 1.0f) {
        std::cerr << "Drop rate must be between 0.0 and 1.0" << std::endl;
        return 1;
    }

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

        // Setup random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Calculate how many vertices to keep
        size_t keepCount = static_cast<size_t>(vertices->count * (1.0f - drop));
        std::vector<bool> keepVertex(vertices->count, false);
        size_t currentKeepCount = 0;

        // Randomly select vertices to keep
        for (size_t i = 0; i < vertices->count && currentKeepCount < keepCount; i++) {
            if (dis(gen) >= drop) {
                keepVertex[i] = true;
                currentKeepCount++;
            }
        }

        // Resize vectors to store only kept vertices
        originalVertices.clear();
        originalColors.clear();
        originalVertices.reserve(keepCount);
        originalColors.reserve(keepCount);

        // Copy only the kept vertices and colors
        for (size_t i = 0; i < vertices->count; i++) {
            if (keepVertex[i]) {
                originalVertices.push_back(make_float3(
                    vertexBuffer[i * 3 + 0],
                    vertexBuffer[i * 3 + 1],
                    vertexBuffer[i * 3 + 2]
                ));
                
                originalColors.push_back(make_float3(
                    colorBuffer[i * 3 + 0] / 255.0f,
                    colorBuffer[i * 3 + 1] / 255.0f,
                    colorBuffer[i * 3 + 2] / 255.0f
                ));
            }
        }

        vertexCount = originalVertices.size();
        std::cout << "Kept " << vertexCount << " points out of " << vertices->count 
                  << " (drop rate: " << drop * 100.0f << "%)" << std::endl;

        // Resize and initialize h_splats with kept vertices
        h_splats.resize(vertexCount);

        for (int i = 0; i < vertexCount; i++) {
            h_splats[i].position = originalVertices[i];
            h_splats[i].scale = make_float3(0.5f, 0.5f, 0.5f);
            h_splats[i].opacity = 0.6f;
            h_splats[i].color = originalColors[i];
            h_splats[i].intensity = 0.0f;
            h_splats[i].rotation = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}