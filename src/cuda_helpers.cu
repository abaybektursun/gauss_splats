#include "cuda_helpers.hpp"


void releaseGPUData(GPUData& data) {
    cudaFree(data.d_image);
    cudaFree(data.d_splats);
    cudaFree(data.d_outSplats);
    cudaFree(data.d_vertices);
    cudaFree(data.d_originalVertices);
    cudaFree(data.d_tileRangeStart);
    cudaFree(data.d_tileRangeEnd);
    cudaFree(data.d_splatCounts);
    cudaFree(data.d_splatOffsets);
    cudaFree(data.d_tileSplats);
    std::cout << "\n";
}