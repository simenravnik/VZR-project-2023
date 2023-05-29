#ifndef MATRIX_CUDA_SK_H
#define MATRIX_CUDA_SK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "../helpers/helper_cuda.h"

__device__ void device_print_matrix(float* mat, int rows, int cols) {
    __syncthreads();
    printf("[");
    for (int i = 0; i < rows * cols; i++) {
        if (i % cols == 0) {
            printf("[");
        }
        printf("%f", mat[i]);
        if (i % cols == cols - 1) {
            printf("]");
        } else {
            printf(", ");
        }
        if ((i + 1) % cols == 0 && i != rows * cols - 1) {
            printf(",\n");
        }
    }
    printf("]\n\n");
}

__device__ void device_matrix_tanh_sk(int idx, float* A, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = tanh(A[idx]);
    }
}

__device__ void device_add_sk(int idx, float* A, float* B, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] + B[idx % colsA];
    }
}

__device__ void device_dot_sk(int idx, float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    __syncthreads();
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

__device__ void device_subtract_sk(int idx, float* A, float* B, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] - B[idx];
    }
}

__device__ void device_hadamard_sk(int idx, float* A, float* B, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * B[idx];
    }
}

__device__ void device_transpose_sk(int idx, float* A, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        int row = idx / colsA;
        int col = idx % colsA;
        C[col * rowsA + row] = A[idx];
    }
}

__device__ void device_sum_sk(int idx, float* A, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < colsA) {
        float sum = 0.0f;
        for (int i = 0; i < rowsA; ++i) {
            sum += A[i * colsA + idx];
        }
        C[idx] = sum;
    }
}

__device__ void device_square_sk(int idx, float* A, float* C, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * A[idx];
    }
}

__device__ void device_scalar_multiply_sk(int idx, float* A, float* C, float scalar, int rowsA, int colsA) {
    __syncthreads();
    if (idx < rowsA * colsA) {
        C[idx] = A[idx] * scalar;
    }
}

__device__ void cuda_compute_H(int idx, float* W1_dev, float* b1_dev, float* Xb_dev, float* H_dev, int batchSize, int features, int hiddenSize) {
    device_dot_sk(idx, Xb_dev, W1_dev, H_dev, batchSize, features, hiddenSize);
    device_add_sk(idx, H_dev, b1_dev, H_dev, batchSize, hiddenSize);
    device_matrix_tanh_sk(idx, H_dev, H_dev, batchSize, hiddenSize);
}

__device__ void cuda_compute_Y_hat(int idx, float* H_dev, float* W2_dev, float* Y_hat_dev, float* b2_dev, int batchSize, int hiddenSize, int outputs) {
    device_dot_sk(idx, H_dev, W2_dev, Y_hat_dev, batchSize, hiddenSize, outputs);
    device_add_sk(idx, Y_hat_dev, b2_dev, Y_hat_dev, batchSize, outputs);
    device_matrix_tanh_sk(idx, Y_hat_dev, Y_hat_dev, batchSize, outputs);
}

__device__ void cuda_compute_E(int idx, float* Y_hat_dev, float* Yb_dev, float* E_dev, int batchSize, int outputs) {
    device_subtract_sk(idx, Y_hat_dev, Yb_dev, E_dev, batchSize, outputs);
}

__device__ void cuda_compute_delta_output(int idx, float* deltaOutput_dev, float* E_dev, float* ones_dev, float* Y_hat_dev, int batchSize, int outputs) {
    device_square_sk(idx, Y_hat_dev, Y_hat_dev, batchSize, outputs);
    device_subtract_sk(idx, ones_dev, Y_hat_dev, deltaOutput_dev, batchSize, outputs);
    device_hadamard_sk(idx, E_dev, deltaOutput_dev, deltaOutput_dev, batchSize, outputs);
}

__device__ void cuda_compute_w2g(int idx, float* W2g_dev, float* H_dev, float* H_transpose_dev, float* deltaOutput_dev, int batchSize, int hiddenSize, int outputs) {
    device_transpose_sk(idx, H_dev, H_transpose_dev, batchSize, hiddenSize);
    device_dot_sk(idx, H_transpose_dev, deltaOutput_dev, W2g_dev, hiddenSize, batchSize, outputs);
}

__device__ void cuda_compute_b2g(int idx, float* b2g_dev, float* deltaOutput_dev, int batchSize, int outputs) {
    device_sum_sk(idx, deltaOutput_dev, b2g_dev, batchSize, outputs);
}

__device__ void cuda_compute_He(int idx, float* He_dev, float* deltaOutput_dev, float* W2_dev, float* W2_transpose_dev, float* H_dev, float* ones2_dev, int batchSize, int outputs, int hiddenSize) {
    device_transpose_sk(idx, W2_dev, W2_transpose_dev, hiddenSize, outputs);
    device_dot_sk(idx, deltaOutput_dev, W2_transpose_dev, He_dev, batchSize, outputs, hiddenSize);

    device_square_sk(idx, H_dev, H_dev, batchSize, hiddenSize);
    device_subtract_sk(idx, ones2_dev, H_dev, H_dev, batchSize, hiddenSize);
    device_hadamard_sk(idx, He_dev, H_dev, He_dev, batchSize, hiddenSize);
}

__device__ void cuda_compute_W1g(int idx, float* W1g_dev, float* Xb_dev, float* Xb_transpose_dev, float* He_dev, int batchSize, int features, int hiddenSize) {
    device_transpose_sk(idx, Xb_dev, Xb_transpose_dev, batchSize, features);
    device_dot_sk(idx, Xb_transpose_dev, He_dev, W1g_dev, features, batchSize, hiddenSize);
}

__device__ void cuda_compute_b1g(int idx, float* b1g_dev, float* He_dev, int batchSize, int hiddenSize) {
    device_sum_sk(idx, He_dev, b1g_dev, batchSize, hiddenSize);
}

__device__ void cuda_update_weights(int idx, float* m, float* g, float eta, int rows, int cols) {
    device_scalar_multiply_sk(idx, g, g, eta, rows, cols);
    device_subtract_sk(idx, m, g, m, rows, cols);
}

__global__ void train_on_gpu(float* W1_dev, float* W2_dev, float* b1_dev, float* b2_dev, float* Xb_dev, float* Yb_dev,
                            float* H_dev, float* Y_hat_dev, float* E_dev, float* deltaOutput_dev, 
                            float* W2g_dev, float* b2g_dev, float* He_dev, float* W1g_dev, float* b1g_dev, 
                            float* ones_dev, float* ones2_dev, float* H_transpose_dev, float* W2_transpose_dev, float* Xb_transpose_dev,
                            int batchSize, int features, int hiddenSize, int outputs, float eta, int maxThreadNeeded) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Forward pass
    cuda_compute_H(idx, W1_dev, b1_dev, Xb_dev, H_dev, batchSize, features, hiddenSize);
    cuda_compute_Y_hat(idx, H_dev, W2_dev, Y_hat_dev, b2_dev, batchSize, hiddenSize, outputs);

    // Backward pass
    cuda_compute_E(idx, Y_hat_dev, Yb_dev, E_dev, batchSize, outputs);
    cuda_compute_delta_output(idx, deltaOutput_dev, E_dev, ones_dev, Y_hat_dev, batchSize, outputs);
    cuda_compute_w2g(idx, W2g_dev, H_dev, H_transpose_dev, deltaOutput_dev, batchSize, hiddenSize, outputs);
    cuda_compute_b2g(idx, b2g_dev, deltaOutput_dev, batchSize, outputs);
    cuda_compute_He(idx, He_dev, deltaOutput_dev, W2_dev, W2_transpose_dev, H_dev, ones2_dev, batchSize, outputs, hiddenSize);
    cuda_compute_W1g(idx, W1g_dev, Xb_dev, Xb_transpose_dev, He_dev, batchSize, features, hiddenSize);
    cuda_compute_b1g(idx, b1g_dev, He_dev, batchSize, hiddenSize);

    // Update weights and biases
    cuda_update_weights(idx, W1_dev, W1g_dev, eta, features, hiddenSize);
    cuda_update_weights(idx, W2_dev, W2g_dev, eta, hiddenSize, outputs);
    cuda_update_weights(idx, b1_dev, b1g_dev, eta, 1, hiddenSize);
    cuda_update_weights(idx, b2_dev, b2g_dev, eta, 1, outputs);
}

#endif