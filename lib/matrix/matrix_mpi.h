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
        return;
    }

    int elements_per_proc = mat1.rows / num_procs;
    int elements_remaining = mat1.rows % num_procs;
    int *send_counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    // Calculate send_counts and displs for scattering the rows of mat1
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = elements_per_proc * mat1.cols;
        if (i < elements_remaining) {
            send_counts[i] += mat1.cols;
        }
        displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
    }

    // Scatter rows of mat1
    float *local_mat1 = malloc(send_counts[rank] * sizeof(float));
    MPI_Scatterv(mat1.data, send_counts, displs, MPI_FLOAT, local_mat1, send_counts[rank], MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    // Calculate partial dot products
    int local_rows = send_counts[rank] / mat1.cols;
    float *local_product = malloc(local_rows * mat2.cols * sizeof(float));

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            double sum = 0;
            for (int k = 0; k < mat1.cols; k++) {
                sum += local_mat1[i * mat1.cols + k] * mat2.data[k * mat2.cols + j];
            }
            local_product[i * mat2.cols + j] = (float)sum;
        }
    }

    int *gather_send_counts = malloc(num_procs * sizeof(int));
    int *gather_displs = malloc(num_procs * sizeof(int));

    for (int i = 0; i < num_procs; i++) {
        gather_send_counts[i] = send_counts[i] / mat1.cols * mat2.cols;
        gather_displs[i] = (i > 0) ? (gather_displs[i - 1] + gather_send_counts[i - 1]) : 0;
    }

    // Gather partial products to the master process
    MPI_Gatherv(local_product, local_rows * mat2.cols, MPI_FLOAT, product.data, gather_send_counts, gather_displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    free(local_mat1);
    free(local_product);
    free(send_counts);
    free(displs);
}

void add_mpi(Matrix mat1, Matrix mat2, int rank, int num_procs) {
    int rows = mat1.rows;
    int cols = mat1.cols;

    int chunk_size = rows / num_procs;
    int extra_rows = rows % num_procs;
    int start_row = rank * chunk_size;
    int end_row = start_row + chunk_size;

    if (rank == num_procs - 1) {
        // Distribute extra rows to the last process
        end_row += extra_rows;
    }

    int* recvcounts = (int*)malloc(num_procs * sizeof(int));
    int* displs = (int*)malloc(num_procs * sizeof(int));

    for (int i = 0; i < num_procs; i++) {
        int chunk_start = i * chunk_size;
        int chunk_end = chunk_start + chunk_size;
        if (i == num_procs - 1) {
            chunk_end += extra_rows;
        }
        int chunk_rows = chunk_end - chunk_start;
        recvcounts[i] = chunk_rows * cols;
        displs[i] = chunk_start * cols;
    }

    // Scatter matrix data to all processes
    MPI_Scatterv(mat1.data, recvcounts, displs, MPI_INT, mat1.data, recvcounts[rank], MPI_INT, MASTER, MPI_COMM_WORLD);

    // Broadcast matrix dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Perform matrix addition in parallel
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            mat1.data[i * cols + j] += mat2.data[j];
        }
    }

    // Gather modified matrix data to the master process
    MPI_Gatherv(mat1.data, recvcounts[rank], MPI_INT, mat1.data, recvcounts, displs, MPI_INT, MASTER, MPI_COMM_WORLD);

    free(recvcounts);
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