#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../lib/read/read.h"
#include "../../../lib/matrix/matrix.h"
#include "../../../lib/helpers/helpers.h"
#include "../../../lib/matrix/matrix_cuda.h"
#include "../../../lib/matrix/matrix_serial.h"

void test_device_add(Matrix A, Matrix B) {

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
    add_serial(A, B);

    // Compare the matrices
    int error = compare_matrices(C, A);

    if (error) {
        print_failed("Device Addition Test");
    } else {
        print_passed("Device Addition Test");
    }

    // Free host memory
    free(C.data);
    free_device_matrix(C_dev);
}

void test_device_dot(Matrix A, Matrix B) {

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
    Matrix C_ref = allocate_matrix(A.rows, B.cols);
    dot_serial(A, B, C_ref);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    if (error) {
        print_failed("Device Dot Test");
    } else {
        print_passed("Device Dot Test");
    }

    // Free host memory
    free(C_ref.data);
    free(C.data);
}

void test_device_tanh(Matrix A) {

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
    
    matrix_tanh_serial(A);

    // Compare the matrices
    int error = compare_matrices(C, A);

    if (error) {
        print_failed("Device Tanh Test");
    } else {
        print_passed("Device Tanh Test");
    }

    // Free host memory
    free(C.data);
}

void test_device_subtract(Matrix A, Matrix B) {

    // Transpose B to get the correct dimensions
    Matrix B_transposed = allocate_matrix(B.cols, B.rows);
    transpose_serial(B, B_transposed);

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
    Matrix C_ref = allocate_matrix(A.rows, A.cols);
    subtract_serial(A, B_transposed, C_ref);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    if (error) {
        print_failed("Device Subtract Test");
    } else {
        print_passed("Device Subtract Test");
    }

    // Free host memory
    free(C_ref.data);
    free(C.data);
}

void test_device_hadamard(Matrix A, Matrix B) {

    // Transpose B to get the correct dimensions
    Matrix B_transposed = allocate_matrix(B.cols, B.rows);
    transpose_serial(B, B_transposed);

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
    Matrix C_ref = allocate_matrix(A.rows, A.cols);
    hadamard_serial(A, B_transposed, C_ref);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    if (error) {
        print_failed("Device Hadamard Test");
    } else {
        print_passed("Device Hadamard Test");
    }


    // Free host memory
    free(C_ref.data);
    free(C.data);
}

void test_device_transpose(Matrix A) {

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
    
    Matrix C_ref = allocate_matrix(A.cols, A.rows);
    transpose_serial(A, C_ref);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);

    if (error) {
        print_failed("Device Transpose Test");
    } else {
        print_passed("Device Transpose Test");
    }

    // Free host memory
    free(C_ref.data);
    free(C.data);
}

void test_device_sum(Matrix A) {

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
    
    Matrix C_ref = allocate_matrix(1, A.cols);
    sum_serial(A, C_ref);

    // Compare the matrices
    int error = compare_matrices(C, C_ref);
    
    if (error) {
        print_failed("Device Sum Test");
    } else {
        print_passed("Device Sum Test");
    }

    // Free host memory
    free(C_ref.data);
    free(C.data);
}

void test_device_square(Matrix A) {

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
    
    square_serial(A);

    // Compare the matrices
    int error = compare_matrices(C, A);

    if (error) {
        print_failed("Device Square Test");
    } else {
        print_passed("Device Square Test");
    }

    // Free host memory
    free(C.data);
}

void test_device_scalar_multiply(Matrix A, float scalar) {
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
    
    scalar_multiply_serial(A, scalar);

    // Compare the matrices
    int error = compare_matrices(C, A);

    if (error) {
        print_failed("Device Scalar Multiply Test");
    } else {
        print_passed("Device Scalar Multiply Test");
    }

    // Free host memory
    free(C.data);
}

int main(int argc, char** argv) {

    const int rows_A = 20;
    const int cols_A = 30;
    const int rows_B = 30;
    const int cols_B = 20;

    Matrix A = random_matrix(rows_A, cols_A);
    Matrix B = random_matrix(rows_B, cols_B);
    Matrix b = random_matrix(1, cols_A);

    test_device_add(A, b);
    test_device_dot(A, B);
    test_device_tanh(A);
    test_device_subtract(A, B);
    test_device_hadamard(A, B);
    test_device_transpose(A);
    test_device_sum(A);
    test_device_square(A);
    test_device_scalar_multiply(A, 28.6);

    return 0;
}