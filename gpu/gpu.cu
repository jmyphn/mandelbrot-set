#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <math.h>

#include "../include/common.hpp"
#include "../include/solver.hpp"

#define uchr unsigned char

#define BLOCK_SIZE 16

int rank, num_procs;
int X, Y, max_iter;
double real_max, real_min, imag_max, imag_min;
uchar *grid;        // on device
uchar *colors;      // on device

void init(uchar* grid_, uchar* colors_, int X_, int Y_, int max_iter_, double real_max_, double real_min_, double imag_max_, double imag_min_, int rank_, int num_procs_) {
    X = X_;
    Y = Y_;
    max_iter = max_iter_;
    real_max = real_max_;
    real_min = real_min_;
    imag_max = imag_max_;
    imag_min = imag_min_;
    rank = rank_;
    num_procs = num_procs_;
    cudaMalloc(&grid, 3 * X * Y * sizeof(uchr));
    cudaMalloc(&colors, (max_iter + 1) * 3 * sizeof(uchr));
    cudaMemset(grid, 0, 3 * X * Y * sizeof(uchr));
    cudaMemcpy(colors, colors_, (max_iter + 1) * 3 * sizeof(uchr), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization error: " << cudaGetErrorString(err) << std::endl;
    }
}

__device__ void calculate_color(uchar* color, const uchar* colors, double i, int max_iter) {
    if (i >= max_iter) {
        color[0] = 0;
        color[1] = 0;
        color[2] = 0;
        return;
    }
    int idx = 3 * (int)i;
    if (idx >= max_iter * 3) idx = (max_iter - 1) * 3;

    uchar color1[3], color2[3];

    color1[0] = colors[idx];
    color1[1] = colors[idx + 1];
    color1[2] = colors[idx + 2];

    idx = idx + 3 > 3 * max_iter ? 3 * max_iter : idx + 3;
    color2[0] = colors[idx];
    color2[1] = colors[idx + 1];
    color2[2] = colors[idx + 2];

    double t = i - (int)i;
    color[0] = static_cast<char>(color1[0] * (1 - t) + color2[0] * t);
    color[1] = static_cast<char>(color1[1] * (1 - t) + color2[1] * t);
    color[2] = static_cast<char>(color1[2] * (1 - t) + color2[2] * t);
}

__global__ void mandelbrot(unsigned char *grid, unsigned char* colors, int X, int Y, int max_iter, double real_max, double real_min, double imag_max, double imag_min) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= X || py >= Y) return;

    double b = 0;
    double a = 0;
    double b0 = real_min + (px * ((real_max - real_min) / (X * 1.0)));
    double a0 = imag_max - (py * ((imag_max - imag_min) / (Y * 1.0)));

    double i = 0;
    double b2 = 0;
    double a2 = 0;

    while (i < max_iter && (a2 + b2) <= 20.) {
        a = 2*b*a + a0;
        b = b2 - a2 + b0;
        b2 = b*b;
        a2 = a*a;
        i++;
    }

    if (i < max_iter) {
        double log_zn = log(a*a + b*b) / 2;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        i += 1.0 - nu;
        if (i < 0) i = 0;
    }

    uchar color[3];
    calculate_color(color, colors, i, max_iter);

    int idx = py * X * 3 + px * 3;
    if (idx + 2 < X * Y * 3) {
        grid[idx] = color[0];
        grid[idx + 1] = color[1];
        grid[idx + 2] = color[2];
    }
    
}

void run() {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((X + dimBlock.x) / dimBlock.x, (Y + dimBlock.y) / dimBlock.y);
    mandelbrot<<<dimGrid, dimBlock>>>(grid, colors, X, Y, max_iter, real_max, real_min, imag_max, imag_min);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl; 
    }
}

void free_memory() {
    cudaFree(grid);
    cudaFree(colors);
}

void transfer_data_to_host(unsigned char *grid_) {
    cudaMemcpy(grid_, grid, 3 * X * Y * sizeof(uchr), cudaMemcpyDeviceToHost);
}

unsigned char* find_mandelbrot(int px, int py) {
    return nullptr;
}

