#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

Matrix create_on_device(int rows, int cols) {
    Matrix m_dev = {
        .rows = rows,
        .cols = cols
    };
    checkCudaErrors(cudaMalloc(&m_dev.data, rows * cols * sizeof(float)));
    return m_dev;
}

Matrix to_device(Matrix m) {
    Matrix m_dev = {
        .rows = m.rows,
        .cols = m.cols
    };
    checkCudaErrors(cudaMalloc(&m_dev.data, m.rows * m.cols * sizeof(float)));
    cudaMemcpy(m_dev.data, m.data, m.rows * m.cols * sizeof(float), cudaMemcpyHostToDevice);
    return m_dev;
}

Matrix to_host(Matrix m) {
    Matrix m_host = {
        .rows = m.rows,
        .cols = m.cols
    };
    m_host.data = (float*) malloc(m.rows * m.cols * sizeof(float));
    cudaMemcpy(m_host.data, m.data, m.rows * m.cols * sizeof(float), cudaMemcpyDeviceToHost);
    return m_host;
}

void free_device_matrix(Matrix m) {
    checkCudaErrors(cudaFree(m.data));
}

// CUDA kernels
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