
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
#include "gs.hpp"
#include "cuda_helpers.hpp"
#include "camera_gpu.cu"

#include <opencv2/opencv.hpp>

cv::Mat normalize_image_for_training(const cv::Mat& input) {
    // 1. Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(input, rgb_img, cv::COLOR_BGR2RGB);
    
    // 2. Convert to float32 and normalize to [0,1]
    cv::Mat float_img;
    rgb_img.convertTo(float_img, CV_32F, 1.0/255.0);

    // Optional: Verify the range
    double min_val, max_val;
    cv::minMaxLoc(float_img, &min_val, &max_val);
    std::cout << "Value range after normalization: [" 
              << min_val << ", " << max_val << "]" << std::endl;
              
    return float_img;
}

cv::Mat convert_pixels_to_normalized(const std::vector<Uint32>& pixels, int width, int height) {
    // Create OpenCV Mat with float values (CV_32FC3)
    cv::Mat normalized(height, width, CV_32FC3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            Uint32 pixel = pixels[idx];
            
            // Extract RGBA values (based on your current format where r is in lowest bits)
            Uint8 r = (pixel >> 0) & 0xFF;   // Red in lowest bits
            Uint8 g = (pixel >> 8) & 0xFF;   // Green
            Uint8 b = (pixel >> 16) & 0xFF;  // Blue
            Uint8 a = (pixel >> 24) & 0xFF;  // Alpha in highest bits
            
            // Convert to float and normalize to [0,1]
            // Note: we're not using alpha in the normalized image
            cv::Vec3f& pixel_normalized = normalized.at<cv::Vec3f>(y, x);
            pixel_normalized[0] = r / 255.0f;  // R
            pixel_normalized[1] = g / 255.0f;  // G
            pixel_normalized[2] = b / 255.0f;  // B
        }
    }
    
    return normalized;
}

// Optional: Function to verify the conversion
void verify_conversion(const cv::Mat& normalized) {
    // Check type
    assert(normalized.type() == CV_32FC3 && "Image should be CV_32FC3");
    
    // Check value range
    double min_val, max_val;
    cv::minMaxLoc(normalized, &min_val, &max_val);
    std::cout << "Value range after normalization: [" 
              << min_val << ", " << max_val << "]" << std::endl;
    
    // Print matrix info
    std::cout << "Channels: " << normalized.channels() << std::endl;
    std::cout << "Depth: " << normalized.depth() << std::endl;
    std::cout << "Type: " << normalized.type() << std::endl;
}

// Usage example:
/*
std::vector<Uint32> pixels = ... // Your pixel data
int width = camera.imageWidth;
int height = camera.imageHeight;

cv::Mat normalized = convert_pixels_to_normalized(pixels, width, height);
verify_conversion(normalized);
*/

int main() {
    cv::Mat img = cv::imread("/workspaces/gauss_splats/mandy.png");
    if (img.empty()) {
        printf("Error loading image\n");
        return -1;
    }

    // Resize if needed (as in your previous code)
    int new_width = 512;
    int new_height = (new_width * img.rows) / img.cols;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_width, new_height));

    // Normalize the image
    cv::Mat normalized = normalize_image_for_training(resized);

    // Now normalized will be:
    // - RGB order
    // - CV_32F type (float)
    // - Values in range [0,1]
    // - Shape: (height, width, 3)

    // You can verify the type
    verify_conversion(normalized);
    
    return 0;
}