#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../../../lib/read/read.h"
#include "../../../lib/helpers/helpers.h"
#include "../../../lib/matrix/matrix.h"
#include "../../../lib/matrix/matrix_serial.h"
#include "../../../lib/matrix/matrix_mpi.h"

#define MASTER 0

void test_mpi_dot(Matrix A, Matrix B, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix D = duplicate_matrix(B);
    Matrix product = allocate_matrix(C.rows, D.cols);
    dot_mpi(C, D, product, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix D_ref = duplicate_matrix(B);
        Matrix product_ref = allocate_matrix(C_ref.rows, D_ref.cols);
        
        dot_serial(C_ref, D_ref, product_ref);

        int error = compare_matrices(product, product_ref);

        if (error) {
            print_failed("MPI Dot Product Test");
        } else {
            print_passed("MPI Dot Product Test");
        }

        free(C_ref.data);
        free(D_ref.data);
        free(product_ref.data);
    }
    free(C.data);
    free(D.data);
    free(product.data);
}

void test_mpi_add(Matrix A, Matrix B, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix D = duplicate_matrix(B);
    add_mpi(C, D, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix D_ref = duplicate_matrix(B);
    
        add_serial(C_ref, D_ref);

        int error = compare_matrices(C, C_ref);

        if (error) {
            print_failed("MPI Add Test");
        } else {
            print_passed("MPI Add Test");
        }

        free(C_ref.data);
        free(D_ref.data);
    }
    free(C.data);
    free(D.data);
}

void test_mpi_subtract(Matrix A, Matrix B, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix D = duplicate_matrix(B);
    Matrix difference = allocate_matrix(C.rows, C.cols);
    subtract_mpi(C, D, difference, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix D_ref = duplicate_matrix(B);
        Matrix difference_ref = allocate_matrix(C_ref.rows, C_ref.cols);
        
        subtract_serial(C_ref, D_ref, difference_ref);

        int error = compare_matrices(difference, difference_ref);

        if (error) {
            print_failed("MPI Subtract Test");
        } else {
            print_passed("MPI Subtract Test");
        }

        free(C_ref.data);
        free(D_ref.data);
        free(difference_ref.data);
    }
    free(C.data);
    free(D.data);
    free(difference.data);
}

void test_mpi_hadamard(Matrix A, Matrix B, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix D = duplicate_matrix(B);
    Matrix product = allocate_matrix(C.rows, C.cols);
    hadamard_mpi(C, D, product, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix D_ref = duplicate_matrix(B);
        Matrix product_ref = allocate_matrix(C_ref.rows, C_ref.cols);
        
        hadamard_serial(C_ref, D_ref, product_ref);

        int error = compare_matrices(product, product_ref);

        if (error) {
            print_failed("MPI Hadamard Test");
        } else {
            print_passed("MPI Hadamard Test");
        }

        free(C_ref.data);
        free(D_ref.data);
        free(product_ref.data);
    }
    free(C.data);
    free(D.data);
    free(product.data);
}

void test_mpi_transpose(Matrix A, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix trans = allocate_matrix(C.rows, C.cols);
    transpose_mpi(C, trans, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix trans_ref = allocate_matrix(C_ref.rows, C_ref.cols);
        
        transpose_serial(C_ref, trans_ref);

        int error = compare_matrices(trans, trans_ref);

        if (error) {
            print_failed("MPI Transpose Test");
        } else {
            print_passed("MPI Transpose Test");
        }

        free(C_ref.data);
        free(trans_ref.data);
    }
    free(C.data);
    free(trans.data);
}

void test_mpi_sum(Matrix A, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    Matrix sum_mat = allocate_matrix(C.rows, C.cols);
    sum_mpi(C, sum_mat, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        Matrix sum_mat_ref = allocate_matrix(C_ref.rows, C_ref.cols);
        
        sum_serial(C_ref, sum_mat_ref);

        int error = compare_matrices(sum_mat, sum_mat_ref);

        if (error) {
            print_failed("MPI Sum Test");
        } else {
            print_passed("MPI Sum Test");
        }

        free(C_ref.data);
        free(sum_mat_ref.data);
    }
    free(C.data);
    free(sum_mat.data);
}

void test_mpi_square(Matrix A, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    square_mpi(C, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);        
        square_serial(C_ref);

        int error = compare_matrices(C, C_ref);

        if (error) {
            print_failed("MPI Square Test");
        } else {
            print_passed("MPI Square Test");
        }

        free(C_ref.data);
    }
    free(C.data);
}

void test_mpi_tanh(Matrix A, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    matrix_tanh_mpi(C, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        matrix_tanh_serial(C_ref);

        int error = compare_matrices(C, C_ref);

        free(C_ref.data);

        if (error) {
            print_failed("MPI Tanh Test");
        } else {
            print_passed("MPI Tanh Test");
        }
    }
    free(C.data);
}

void test_mpi_scalar_multiply(Matrix A, float scalar, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    scalar_multiply_mpi(C, scalar, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        
        scalar_multiply_serial(C_ref, scalar);

        int error = compare_matrices(C, C_ref);

        if (error) {
            print_failed("MPI Scalar Multiply Test");
        } else {
            print_passed("MPI Scalar Multiply Test");
        }

        free(C_ref.data);
    }
    free(C.data);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rows_A = 20;
    const int cols_A = 30;
    const int rows_B = 30;
    const int cols_B = 20;
    const float scalar = 28.6;

    Matrix A = random_matrix(rows_A, cols_A);
    Matrix B = random_matrix(rows_B, cols_B);
    Matrix b = random_matrix(1, cols_A);

    test_mpi_dot(A, B, rank, num_procs);
    test_mpi_add(A, b, rank, num_procs);
    test_mpi_subtract(A, B, rank, num_procs);
    test_mpi_hadamard(A, B, rank, num_procs);
    test_mpi_transpose(A, rank, num_procs);
    test_mpi_sum(A, rank, num_procs);
    test_mpi_square(A, rank, num_procs);
    test_mpi_tanh(A, rank, num_procs);
    test_mpi_scalar_multiply(A, scalar, rank, num_procs);

    MPI_Finalize();

    return 0;
}