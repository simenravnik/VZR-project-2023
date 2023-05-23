#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../lib/read.h"
#include "../../../lib/matrix.h"
#include "../../../lib/helpers.h"
#include "../../../lib/cuda_matrix.h"
#include "colors.h"

void print_failed(const char* test_name) {
    printf("%27s: ", test_name);
    red();
    printf("failed\n");
    reset();
}

void print_passed(const char* test_name) {
    printf("%27s: ", test_name);
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

    device_add<<<gridSize, blockSize>>>(A_dev.data, B_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

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

int test_device_subtract(Matrix A, Matrix B) {

    // Transpose B to get the correct dimensions
    Matrix B_transposed = transpose(B);

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix B_dev = to_device(B_transposed);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * B_dev.cols + blockSize - 1) / blockSize;

    device_subtract<<<gridSize, blockSize>>>(A_dev.data, B_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(B_dev);
    free_device_matrix(C_dev);

    // Check if the dot product is correct
    Matrix C_ref = subtract(A, B_transposed);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_hadamard(Matrix A, Matrix B) {

    // Transpose B to get the correct dimensions
    Matrix B_transposed = transpose(B);

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix B_dev = to_device(B_transposed);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * B_dev.cols + blockSize - 1) / blockSize;

    device_hadamard<<<gridSize, blockSize>>>(A_dev.data, B_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(B_dev);
    free_device_matrix(C_dev);

    // Check if the dot product is correct
    Matrix C_ref = hadamard(A, B_transposed);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_transpose(Matrix A) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix C_dev = create_on_device(A.cols, A.rows);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_transpose<<<gridSize, blockSize>>>(A_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(C_dev);
    
    Matrix C_ref = transpose(A);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_sum(Matrix A) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix C_dev = create_on_device(1, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_sum<<<gridSize, blockSize>>>(A_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(C_dev);
    
    Matrix C_ref = sum(A);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_square(Matrix A) {

    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_square<<<gridSize, blockSize>>>(A_dev.data, C_dev.data, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(C_dev);
    
    Matrix C_ref = square(A);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    // Free host memory
    free(C_ref.data);
    free(C.data);

    return error;
}

int test_device_scalar_multiply(Matrix A, float scalar) {
    // Allocate memory on the device
    Matrix A_dev = to_device(A);
    Matrix C_dev = create_on_device(A.rows, A.cols);

    // Block size and grid size
    int blockSize = 32;
    int gridSize = (A_dev.rows * A_dev.cols + blockSize - 1) / blockSize;

    device_scalar_multiply<<<gridSize, blockSize>>>(A_dev.data, C_dev.data, scalar, A_dev.rows, A_dev.cols);

    // Copy data back to the host
    Matrix C = to_host(C_dev);

    // Free device memory
    free_device_matrix(A_dev);
    free_device_matrix(C_dev);
    
    Matrix C_ref = scalar_multiply(A, scalar);

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

    // Subtract of A and B
    if (test_device_subtract(A, B)) {
        print_failed("Device Subtract Test");
    } else {
        print_passed("Device Subtract Test");
    }

    // Hadamard of A and B
    if (test_device_hadamard(A, B)) {
        print_failed("Device Hadamard Test");
    } else {
        print_passed("Device Hadamard Test");
    }

    // Transpose
    if (test_device_transpose(A)) {
        print_failed("Device Transpose Test");
    } else {
        print_passed("Device Transpose Test");
    }

    // Sum
    if (test_device_sum(A)) {
        print_failed("Device Sum Test");
    } else {
        print_passed("Device Sum Test");
    }

    // Square
    if (test_device_square(A)) {
        print_failed("Device Square Test");
    } else {
        print_passed("Device Square Test");
    }

    // Scalar Multiply
    if (test_device_scalar_multiply(A, 28.6)) {
        print_failed("Device Scalar Multiply Test");
    } else {
        print_passed("Device Scalar Multiply Test");
    }

    return 0;
}