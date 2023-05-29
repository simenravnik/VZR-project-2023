#ifndef MATRIX_MPI_H
#define MATRIX_MPI_H
#define MASTER 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "matrix.h"

void dot_mpi(Matrix mat1, Matrix mat2, Matrix product, int rank, int num_procs);
void add_mpi(Matrix mat1, Matrix mat2, int rank, int num_procs);
void subtract_mpi(Matrix mat1, Matrix mat2, Matrix difference, int rank, int num_procs);
void hadamard_mpi(Matrix mat1, Matrix mat2, Matrix product, int rank, int num_procs);
void transpose_mpi(Matrix mat, Matrix trans, int rank, int num_procs);
void sum_mpi(Matrix mat, Matrix sum, int rank, int num_procs);
void square_mpi(Matrix mat, int rank, int num_procs);
void matrix_tanh_mpi(Matrix mat, int rank, int num_procs);
void scalar_multiply_mpi(Matrix mat, float scalar, int rank, int num_procs);

void dot_mpi(Matrix mat1, Matrix mat2, Matrix product, int rank, int num_procs) {
    if (mat1.cols != mat2.rows) {
        if (rank == MASTER) {
            printf("Error: Matrix dimensions do not match for dot product\n");
        }
        MPI_Finalize();
        exit(1);
    }

    int rows_per_proc = mat1.rows / num_procs;
    int rows_remaining = mat1.rows % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the matrices
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = rows_per_proc * mat1.cols;
        if (i < rows_remaining) {
            send_counts[i] += mat1.cols;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter matrices mat1 and mat2
    float *local_mat1 = malloc(send_counts[rank] * sizeof(float));
    float *local_mat2 = malloc(mat2.rows * mat2.cols * sizeof(float));
    MPI_Scatterv(mat1.data, send_counts, displs, MPI_FLOAT, local_mat1, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(mat2.data, mat2.rows * mat2.cols, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Compute local dot product
    float *local_product = malloc(send_counts[rank] * mat2.cols * sizeof(float));
    int local_rows = send_counts[rank] / mat1.cols;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            double sum = 0;
            for (int k = 0; k < mat1.cols; k++) {
                sum += local_mat1[i * mat1.cols + k] * mat2.data[k * mat2.cols + j];
            }
            local_product[i * mat2.cols + j] = (float)sum;
        }
    }

    // Gather local products to the master process
    MPI_Gatherv(local_product, send_counts[rank], MPI_FLOAT, product.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat1);
    free(local_mat2);
    free(local_product);
    free(send_counts);
    free(displs);
}

void add_mpi(Matrix mat1, Matrix mat2, int rank, int num_procs) {
    int cols_per_proc = mat1.cols / num_procs;
    int cols_remaining = mat1.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the columns
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = cols_per_proc;
        if (i < cols_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter columns of mat1 and broadcast mat2
    float *local_mat1 = malloc(mat1.rows * send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat1.data, send_counts, displs, MPI_FLOAT, local_mat1, mat1.rows * send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(mat2.data, mat2.rows * mat2.cols, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Add mat2 to local columns of mat1
    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < send_counts[rank]; j++) {
            int col_index = displs[rank] + j;
            local_mat1[i * send_counts[rank] + j] += mat2.data[i * mat2.cols + col_index];
        }
    }

    // Gather the updated columns to the master process
    MPI_Gatherv(local_mat1, mat1.rows * send_counts[rank], MPI_FLOAT, mat1.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat1);
    free(send_counts);
    free(displs);
}

void subtract_mpi(Matrix mat1, Matrix mat2, Matrix difference, int rank, int num_procs) {
    int elements_per_proc = mat1.rows * mat1.cols / num_procs;
    int elements_remaining = mat1.rows * mat1.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the elements
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc;
        if (i < elements_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter elements of mat1 and mat2
    float *local_mat1 = malloc(send_counts[rank] * sizeof(float));
    float *local_mat2 = malloc(send_counts[rank] * sizeof(float));
    float *local_difference = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat1.data, send_counts, displs, MPI_FLOAT, local_mat1, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(mat2.data, send_counts, displs, MPI_FLOAT, local_mat2, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate local differences
    for (int i = 0; i < send_counts[rank]; i++) {
        local_difference[i] = local_mat1[i] - local_mat2[i];
    }

    // Gather the local differences to the master process
    MPI_Gatherv(local_difference, send_counts[rank], MPI_FLOAT, difference.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat1);
    free(local_mat2);
    free(local_difference);
    free(send_counts);
    free(displs);
}

void hadamard_mpi(Matrix mat1, Matrix mat2, Matrix product, int rank, int num_procs) {
    int elements_per_proc = mat1.rows * mat1.cols / num_procs;
    int elements_remaining = mat1.rows * mat1.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the elements
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc;
        if (i < elements_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter elements of mat1 and mat2
    float *local_mat1 = malloc(send_counts[rank] * sizeof(float));
    float *local_mat2 = malloc(send_counts[rank] * sizeof(float));
    float *local_product = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat1.data, send_counts, displs, MPI_FLOAT, local_mat1, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(mat2.data, send_counts, displs, MPI_FLOAT, local_mat2, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate local product
    for (int i = 0; i < send_counts[rank]; i++) {
        local_product[i] = local_mat1[i] * local_mat2[i];
    }

    // Gather the local products to the master process
    MPI_Gatherv(local_product, send_counts[rank], MPI_FLOAT, product.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat1);
    free(local_mat2);
    free(local_product);
    free(send_counts);
    free(displs);
}

void transpose_mpi(Matrix mat, Matrix trans, int rank, int num_procs) {
    int elements_per_proc = mat.rows * mat.cols / num_procs;
    int elements_remaining = mat.rows * mat.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the elements
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc;
        if (i < elements_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter elements of mat
    float *local_mat = malloc(send_counts[rank] * sizeof(float));
    float *local_trans = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat.data, send_counts, displs, MPI_FLOAT, local_mat, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate local transposition
    for (int i = 0; i < send_counts[rank]; i++) {
        int row = (displs[rank] + i) % mat.rows;
        int col = (displs[rank] + i) / mat.rows;
        int trans_index = col * mat.rows + row;
        local_trans[i] = local_mat[i];
    }

    // Gather the local transpositions to the master process
    MPI_Gatherv(local_trans, send_counts[rank], MPI_FLOAT, trans.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat);
    free(local_trans);
    free(send_counts);
    free(displs);
}

void sum_mpi(Matrix mat, Matrix sum, int rank, int num_procs) {
    int cols_per_proc = mat.cols / num_procs;
    int cols_remaining = mat.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the columns
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = cols_per_proc;
        if (i < cols_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter columns of mat
    float *local_mat = malloc(mat.rows * send_counts[rank] * sizeof(float));
    float *local_sum = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat.data, send_counts, displs, MPI_FLOAT, local_mat, mat.rows * send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate local column sums
    for (int i = 0; i < send_counts[rank]; i++) {
        double colSum = 0;
        for (int j = 0; j < mat.rows; j++) {
            colSum += local_mat[j * send_counts[rank] + i];
        }
        local_sum[i] = (float)colSum;
    }

    // Gather the local column sums to the master process
    MPI_Gatherv(local_sum, send_counts[rank], MPI_FLOAT, sum.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat);
    free(local_sum);
    free(send_counts);
    free(displs);
}

void square_mpi(Matrix mat, int rank, int num_procs) {
    int num_elements = mat.rows * mat.cols;
    int local_elements = num_elements / num_procs;
    int remaining_elements = num_elements % num_procs;

    int* send_counts = malloc(num_procs * sizeof(int));
    int* displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the matrix
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = local_elements;
        if (i < remaining_elements) {
            send_counts[i]++;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter matrix data
    float* local_data = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat.data, send_counts, displs, MPI_FLOAT, local_data, send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Square the local elements
    for (int i = 0; i < send_counts[rank]; i++) {
        local_data[i] = local_data[i] * local_data[i];
    }

    // Gather squared elements to the root process
    MPI_Gatherv(local_data, send_counts[rank], MPI_FLOAT, mat.data, send_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(send_counts);
    free(displs);
    free(local_data);
}

void matrix_tanh_mpi(Matrix mat, int rank, int num_procs) {
    int elements_per_proc = mat.rows * mat.cols / num_procs;
    int elements_remaining = mat.rows * mat.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the elements
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc;
        if (i < elements_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter elements of mat
    float *local_mat = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat.data, send_counts, displs, MPI_FLOAT, local_mat, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate local hyperbolic tangent
    for (int i = 0; i < send_counts[rank]; i++) {
        local_mat[i] = tanhf(local_mat[i]);
    }

    // Gather the local results to the master process
    MPI_Gatherv(local_mat, send_counts[rank], MPI_FLOAT, mat.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat);
    free(send_counts);
    free(displs);
}

void scalar_multiply_mpi(Matrix mat, float scalar, int rank, int num_procs) {
    int elements_per_proc = mat.rows * mat.cols / num_procs;
    int elements_remaining = mat.rows * mat.cols % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the elements
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc;
        if (i < elements_remaining) {
            send_counts[i] += 1;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter elements of mat
    float *local_mat = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat.data, send_counts, displs, MPI_FLOAT, local_mat, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Multiply local elements by scalar
    for (int i = 0; i < send_counts[rank]; i++) {
        local_mat[i] = local_mat[i] * scalar;
    }

    // Gather the local results to the master process
    MPI_Gatherv(local_mat, send_counts[rank], MPI_FLOAT, mat.data, send_counts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat);
    free(send_counts);
    free(displs);
}

#endif