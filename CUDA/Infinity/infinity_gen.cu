#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 1920
#define HEIGHT 1080
#define CHANNELS 3

// Corrected hash function for CUDA
__device__ float fract(float x) {
    return x - floorf(x);
}

__device__ float hash(float n) {
    return fract(sinf(n) * 43758.5453123f);
}

// Parametric Infinity (Lemniscate of Gerono)
__device__ void getInfinityPos(float t, float& x, float& y) {
    x = sinf(t);
    y = sinf(t) * cosf(t);
}

__global__ void renderKernel(unsigned char* ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    float u = (float(x) / WIDTH) * 2.0f - 1.0f;
    float v = (float(y) / HEIGHT) * 2.0f - 1.0f;
    v *= (float(HEIGHT) / WIDTH);

    // 1. SPACE BACKGROUND
    // Using the custom fract and sinf
    float n = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    float noise = n - floorf(n); 
    float star = (noise > 0.998f) ? powf(noise, 20.0f) : 0.0f;
    
    float r = star, g = star, b = star;

    // 2. INFINITY SYMBOL GLOW
    float minDist = 1e10;
    const int samples = 128;
    for (int i = 0; i < samples; i++) {
        float t = 6.28318f * (float(i) / samples);
        float ix = 0.8f * sinf(t);
        float iy = 0.8f * sinf(t) * cosf(t);

        float dx = u - ix;
        float dy = v - iy;
        float dist = dx*dx + dy*dy;
        if (dist < minDist) minDist = dist;
    }
    
    minDist = sqrtf(minDist);
    float glow = 0.015f / (minDist + 0.005f);
    
    // 3. COLORING
    r += glow * (0.5f + 0.5f * cosf(u + 0.0f));
    g += glow * (0.5f + 0.5f * cosf(u + 2.0f));
    b += glow * (0.5f + 0.5f * cosf(u + 4.0f));

    int offset = (y * WIDTH + x) * CHANNELS;
    ptr[offset + 0] = (unsigned char)(fminf(r, 1.0f) * 255);
    ptr[offset + 1] = (unsigned char)(fminf(g, 1.0f) * 255);
    ptr[offset + 2] = (unsigned char)(fminf(b, 1.0f) * 255);
}

int main() {
    size_t size = WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char);
    unsigned char *d_ptr, *h_ptr;

    h_ptr = (unsigned char*)malloc(size);
    cudaMalloc(&d_ptr, size);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    std::cout << "Rendering on GPU..." << std::endl;
    renderKernel<<<gridSize, blockSize>>>(d_ptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

    std::cout << "Saving images..." << std::endl;
    stbi_write_png("infinity.png", WIDTH, HEIGHT, CHANNELS, h_ptr, WIDTH * CHANNELS);
    stbi_write_jpg("infinity.jpg", WIDTH, HEIGHT, CHANNELS, h_ptr, 90);

    std::cout << "Done! Files saved as infinity.png and infinity.jpg" << std::endl;

    free(h_ptr);
    cudaFree(d_ptr);
    return 0;
}
