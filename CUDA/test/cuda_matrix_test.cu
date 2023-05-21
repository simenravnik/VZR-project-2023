#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/read.h"
#include "../lib/matrix.h"
#include "../lib/helpers.h"
#include "../lib/cuda_matrix.h"
#include "colors.h"

void print_failed(const char* test_name) {
    printf("%s: ", test_name);
    red();
    printf("failed\n");
    reset();
}

void print_passed(const char* test_name) {
    printf("%s: ", test_name);
    green();
    printf("passed\n");
    reset();
}

int test_device_dot(float* A, float* B, int rowsA, int colsA, int colsB) {

    float* C = allocate_matrix(rowsA, colsB);

    float* A_dev;
    float* B_dev;
    float* C_dev;

    checkCudaErrors(cudaMalloc(&A_dev, rowsA * colsA * sizeof(float)));
    checkCudaErrors(cudaMalloc(&B_dev, colsA * colsB * sizeof(float)));
    checkCudaErrors(cudaMalloc(&C_dev, rowsA * colsB * sizeof(float)));

    // Transfer the image from the host to the device
    cudaMemcpy(A_dev, A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (rowsA * colsB + blockSize - 1) / blockSize;

    device_dot<<<gridSize, blockSize>>>(A_dev, B_dev, C_dev, rowsA, colsA, colsB);

    // Copy data back to the host
    checkCudaErrors(cudaMemcpy(C, C_dev, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    checkCudaErrors(cudaFree(A_dev));
    checkCudaErrors(cudaFree(B_dev));
    checkCudaErrors(cudaFree(C_dev));

    // Check if the dot product is correct
    float* C_ref = dot(A, B, rowsA, colsA, colsA, colsB);

    int error = 0;
    for (int i = 0; i < rowsA * colsB; i++) {
        if (abs(C[i] - C_ref[i]) > 0.0001) {
            error = 1;
            break;
        }
    }

    free(C_ref);
    free(C);

    return error;
}

int main(int argc, char** argv) {

    const int rows_A = 10;
    const int cols_A = 20;
    const int rows_B = 20;
    const int cols_B = 10;

    float* A = random_matrix(rows_A, cols_A);
    float* B = random_matrix(rows_B, cols_B);
    
    // Dot product of A and B
    if (test_device_dot(A, B, rows_A, cols_A, cols_B)) {
        print_failed("Device Dot Test");
    } else {
        print_passed("Device Dot Test");
    }

    return 0;
}