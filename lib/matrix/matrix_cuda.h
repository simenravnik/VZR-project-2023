#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "matrix.h"
#include "../helpers/helper_cuda.h"

Matrix create_on_device(int rows, int cols);
Matrix to_device(Matrix m);
Matrix to_host(Matrix m);
void free_device_matrix(Matrix m);

// CUDA kernels
__global__ void device_matrix_tanh(float* A, float* C, int rowsA, int colsA);
__global__ void device_add(float* A, float* B, float* C, int rowsA, int colsA);
__global__ void device_dot(float* A, float* B, float* C, int rowsA, int colsA, int colsB);
__global__ void device_subtract(float* A, float* B, float* C, int rowsA, int colsA);
__global__ void device_hadamard(float* A, float* B, float* C, int rowsA, int colsA);
__global__ void device_transpose(float* A, float* C, int rowsA, int colsA);
__global__ void device_sum(float* A, float* C, int rowsA, int colsA);
__global__ void device_square(float* A, float* C, int rowsA, int colsA);
__global__ void device_scalar_multiply(float* A, float* C, float scalar, int rowsA, int colsA);

#endif