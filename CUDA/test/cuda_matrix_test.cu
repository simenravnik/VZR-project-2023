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

int test_device_dot(Matrix A, Matrix B) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix B_dev = to_device(B);
    Matrix C_dev = create_on_device(A.rows, B.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * B_dev.cols + blockSize - 1) / blockSize;

    device_dot<<<gridSize, blockSize>>>(A_dev.data, B_dev.data, C_dev.data, A_dev.rows, A_dev.cols, B_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_matrix(A_dev);
    free_matrix(B_dev);
    free_matrix(C_dev);

    // Check if the dot product is correct
    float* C_ref_data = dot(A.data, B.data, A.rows, A.cols, B.rows, B.cols);
    Matrix C_ref = { C_ref_data, A.rows, B.cols };

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int main(int argc, char** argv) {

    const int rows_A = 10;
    const int cols_A = 20;
    const int rows_B = 20;
    const int cols_B = 10;

    Matrix A = create_random_matrix(rows_A, cols_A);
    Matrix B = create_random_matrix(rows_B, cols_B);
    
    // Dot product of A and B
    if (test_device_dot(A, B)) {
        print_failed("Device Dot Test");
    } else {
        print_passed("Device Dot Test");
    }

    return 0;
}