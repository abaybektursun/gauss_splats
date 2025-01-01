#include "camera.hpp"
#include <iostream>

void printCameraParams(const OrthoCameraParams& cam) {
    std::cout << "Camera params:\n"
              << "  xMin=" << cam.xMin << ", xMax=" << cam.xMax << "\n"
              << "  yMin=" << cam.yMin << ", yMax=" << cam.yMax << "\n"
              << "  imageWidth=" << cam.imageWidth
              << ", imageHeight=" << cam.imageHeight
              << std::endl;
}

// If you have more CPU-side camera operations, define them here.
