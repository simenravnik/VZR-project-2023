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

void test_mpi_tanh(Matrix A, int rank, int num_procs) {

    Matrix C = duplicate_matrix(A);
    matrix_tanh_mpi(C, rank, num_procs);

    if (rank == MASTER) {

        Matrix C_ref = duplicate_matrix(A);
        matrix_tanh_serial(C_ref);

        int error = compare_matrices(C, C_ref);

        print_matrix(C);
        print_matrix(C_ref);

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

    A.data[0] = 1;
    A.data[1] = 2;
    A.data[2] = 3;
    A.data[3] = 4;
    A.data[4] = 5;
    A.data[5] = 6;

    // Tanh of A
    test_mpi_tanh(A, rank, num_procs);

    MPI_Finalize();

    return 0;
}