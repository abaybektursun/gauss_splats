#pragma once

/**
 * @brief Stores parameters for an orthographic camera.
 */
struct OrthoCameraParams {
    float xMin, xMax;
    float yMin, yMax;
    int   imageWidth;
    int   imageHeight;
};

/**
 * @brief Example CPU-side function to print camera parameters.
 *        Declared here, defined in camera.cpp.
 */
void printCameraParams(const OrthoCameraParams& cam);

