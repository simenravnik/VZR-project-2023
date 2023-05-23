#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix_cuda.h"

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
__global__ void device_matrix_tanh(float* A, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = tanh(A[idx]);
    }
}

__global__ void device_add(float* A, float* B, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] + B[idx % colsA];
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

__global__ void device_subtract(float* A, float* B, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void device_hadamard(float* A, float* B, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void device_transpose(float* A, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        int row = idx / colsA;
        int col = idx % colsA;
        C[col * rowsA + row] = A[idx];
    }
}

__global__ void device_sum(float* A, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < colsA) {
        float sum = 0.0f;
        for (int i = 0; i < rowsA; ++i) {
            sum += A[i * colsA + idx];
        }
        C[idx] = sum;
    }
}

__global__ void device_square(float* A, float* C, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * A[idx];
    }
}

__global__ void device_scalar_multiply(float* A, float* C, float scalar, int rowsA, int colsA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * scalar;
    }
}