#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/read.h"
#include "../lib/matrix.h"
#include "../lib/helpers.h"
#include "../lib/cuda_matrix.h"
#include "colors.h"

void print_failed(const char* test_name) {
    printf("%20s: ", test_name);
    red();
    printf("failed\n");
    reset();
}

void print_passed(const char* test_name) {
    printf("%20s: ", test_name);
    green();
    printf("passed\n");
    reset();
}

int test_device_add(Matrix A, Matrix B) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix B_dev = to_device(B);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_add<<<gridSize, blockSize>>>(A_dev.data, B_dev.data, C_dev.data, A_dev.rows, A_dev.cols, B_dev.rows, B_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(B_dev);

    // Check if the addition is correct
    Matrix C_ref = add(A, B);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);
    free_device_matrix(C_dev);

    return error;
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
    free_device_matrix(A_dev);
    free_device_matrix(B_dev);
    free_device_matrix(C_dev);

    // Check if the dot product is correct
    Matrix C_ref = dot(A, B);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_tanh(Matrix A) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_matrix_tanh<<<gridSize, blockSize>>>(A_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(C_dev);
    
    Matrix C_ref = matrix_tanh(A);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int main(int argc, char** argv) {

    const int rows_A = 20;
    const int cols_A = 30;
    const int rows_B = 30;
    const int cols_B = 20;

    Matrix A = random_matrix(rows_A, cols_A);
    Matrix B = random_matrix(rows_B, cols_B);
    Matrix b = random_matrix(1, cols_A);

    A.data[0] = 1;
    A.data[1] = 2;
    A.data[2] = 3;
    A.data[3] = 4;
    A.data[4] = 5;
    A.data[5] = 6;

    // Add of A and B
    if (test_device_add(A, b)) {
        print_failed("Device Addition Test");
    } else {
        print_passed("Device Addition Test");
    }
    
    // Dot product of A and B
    if (test_device_dot(A, B)) {
        print_failed("Device Dot Test");
    } else {
        print_passed("Device Dot Test");
    }

    // Tanh of A
    if (test_device_tanh(A)) {
        print_failed("Device Tanh Test");
    } else {
        print_passed("Device Tanh Test");
    }

    return 0;
}