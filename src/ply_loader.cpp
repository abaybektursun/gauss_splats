#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief Example function that reads a PLY file and returns vertices & faces.
 *        You could also define a small struct PLYData to encapsulate them.
 */
bool loadPlyFile(const std::string& filePath,
                 std::vector<float3>& outVertices,
                 std::vector<int>& outFaces,
                 int& vertexCount, int& faceCount)
{
    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Failed to open " << filePath << "\n";
        return false;
    }

    std::string line;
    vertexCount = 0;
    faceCount   = 0;

    // Read header lines
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> vertexCount; // e.g. "element vertex 1234"
        } 
        else if (line.find("element face") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> faceCount;
        } 
        else if (line.find("end_header") != std::string::npos) {
            // Stop reading the header
            break;
        }
    }

    // Read vertices
    outVertices.reserve(vertexCount);
    for (int i = 0; i < vertexCount; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Failed to read line for vertex " << i << "\n";
            return false;
        }
        float x, y, z;
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z)) {
            std::cerr << "Failed to parse x,y,z\n";
            return false;
        }
        outVertices.push_back(make_float3(x, y, z));
    }

    // Read face info (if the file has face data)
    // This can vary across PLY files. For demonstration:
    int tmp;
    outFaces.reserve(faceCount * 3); // e.g. if each face is a triangle
    for (int i = 0; i < faceCount; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Failed to read line for face " << i << "\n";
            return false;
        }
        std::istringstream iss(line);
        int vertexPerFace;
        iss >> vertexPerFace; // typically 3 or 4
        for (int j = 0; j < vertexPerFace; j++) {
            iss >> tmp;
            outFaces.push_back(tmp);
        }
    }

    return true;
}
