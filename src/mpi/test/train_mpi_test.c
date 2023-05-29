#include "/usr/include/openmpi-x86_64/mpi.h"
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

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rows_A = 2;
    const int cols_A = 3;
    const int rows_B = 3;
    const int cols_B = 2;

    Matrix A = random_matrix(rows_A, cols_A);
    Matrix B = random_matrix(rows_B, cols_B);
    Matrix b = random_matrix(1, cols_A);

    test_mpi_tanh(A, rank, num_procs);
    test_mpi_subtract(A, B, rank, num_procs);

    MPI_Finalize();

    return 0;
}