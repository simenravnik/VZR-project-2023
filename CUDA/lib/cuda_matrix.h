#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

__global__ void device_matrix_tanh(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = tanh(input[idx]);
    }
}

__global__ void device_add(float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] += B[idx];
    }
}

__global__ void device_dot(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rowsA * colsB) {
        int row = idx / colsB;
        int col = idx % colsB;
        
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[idx] = sum;
    }
}

#endif