#ifndef TRAIN_MPI_H
#define TRAIN_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../../lib/matrix/matrix.h"
#include "../../lib/matrix/matrix_mpi.h"
#include "../../lib/matrix/matrix_serial.h"
#include "../../lib/models/mlp_model.h"

#define MASTER 0

void mpi_compute_H(Matrix H, Matrix Xb, Matrix W1, Matrix b1, int rank, int num_procs) {
    dot_mpi(Xb, W1, H, rank, num_procs);
    add_serial(H, b1);  // Fix this
    matrix_tanh_mpi(H, rank, num_procs);
}

void mpi_compute_Y_hat(Matrix Y_hat, Matrix H, Matrix W2, Matrix b2, int rank, int num_procs) {
    dot_mpi(H, W2, Y_hat, rank, num_procs);
    add_serial(Y_hat, b2);  // Fix this
    matrix_tanh_mpi(Y_hat, rank, num_procs);
}

void mpi_compute_E(Matrix E, Matrix Y_hat, Matrix Yb, int rank, int num_procs) {
    subtract_mpi(Y_hat, Yb, E, rank, num_procs);
}

void mpi_compute_delta_output(Matrix deltaOutput, Matrix E, Matrix Y_hat, Matrix ones_matrix, int rank, int num_procs) {
    square_mpi(Y_hat, rank, num_procs);
    subtract_mpi(ones_matrix, Y_hat, Y_hat, rank, num_procs);
    hadamard_mpi(E, Y_hat, deltaOutput, rank, num_procs);
}

void mpi_compute_W2g(Matrix W2g, Matrix H, Matrix H_tranpose, Matrix deltaOutput, int rank, int num_procs) {
    transpose_mpi(H, H_tranpose, rank, num_procs);
    dot_mpi(H_tranpose, deltaOutput, W2g, rank, num_procs);
}

void mpi_compute_b2g(Matrix b2g, Matrix deltaOutput, int rank, int num_procs) {
    sum_mpi(deltaOutput, b2g, rank, num_procs);
}

void mpi_compute_He(Matrix He, Matrix deltaOutput, Matrix W2, Matrix W2_transpose, Matrix H, Matrix ones2_matrix, int rank, int num_procs) {
    transpose_mpi(W2, W2_transpose, rank, num_procs);
    dot_mpi(deltaOutput, W2_transpose, He, rank, num_procs);
    square_mpi(H, rank, num_procs);
    subtract_mpi(ones2_matrix, H, H, rank, num_procs);
    hadamard_mpi(He, H, He, rank, num_procs);
}

void mpi_compute_W1g(Matrix W1g, Matrix Xb, Matrix Xb_transpose, Matrix He, int rank, int num_procs) {
    transpose_mpi(Xb, Xb_transpose, rank, num_procs);
    dot_mpi(Xb_transpose, He, W1g, rank, num_procs);
}

void mpi_compute_b1g(Matrix b1g, Matrix He, int rank, int num_procs) {
    sum_mpi(He, b1g, rank, num_procs);
}

void mpi_update_weights(Matrix m, Matrix g, float eta, int rank, int num_procs) {
    scalar_multiply_mpi(g, eta, rank, num_procs);
    subtract_mpi(m, g, m, rank, num_procs);
}

MLP_model train_mpi(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs, int rank, int num_procs);

MLP_model train_mpi(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs, int rank, int num_procs) {

    int samples = X.rows;
    int features = X.cols;
    int outputs = Y.cols;

    // Initialize weights and biases
    Matrix W1 = random_matrix(features, hiddenSize);
    Matrix W2 = random_matrix(hiddenSize, outputs);
    Matrix b1 = random_matrix(1, hiddenSize);
    Matrix b2 = random_matrix(1, outputs);

    Matrix H = allocate_matrix(batchSize, hiddenSize);
    Matrix Y_hat = allocate_matrix(batchSize, outputs);
    Matrix E = allocate_matrix(batchSize, outputs);
    Matrix deltaOutput = allocate_matrix(batchSize, outputs);
    Matrix W2g = allocate_matrix(hiddenSize, outputs);
    Matrix b2g = allocate_matrix(1, outputs);
    Matrix He = allocate_matrix(batchSize, hiddenSize);
    Matrix W1g = allocate_matrix(features, hiddenSize);
    Matrix b1g = allocate_matrix(1, hiddenSize);
    Matrix ones_matrix = ones(batchSize, outputs);
    Matrix ones2_matrix = ones(batchSize, hiddenSize);
    Matrix H_transpose = allocate_matrix(hiddenSize, batchSize);
    Matrix W2_transpose = allocate_matrix(outputs, hiddenSize);
    Matrix Xb_transpose = allocate_matrix(features, batchSize);

    Matrix Xb, Yb;

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int batch = 0; batch < samples; batch += batchSize) {
            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            // Forward pass
            mpi_compute_H(H, Xb, W1, b1, rank, num_procs);   // batchSize x hiddenSize
            mpi_compute_Y_hat(Y_hat, H, W2, b2, rank, num_procs);    // batchSize x outputs

            // Backward pass
            mpi_compute_E(E, Y_hat, Yb, rank, num_procs);    // batchSize x outputs
            mpi_compute_delta_output(deltaOutput, E, Y_hat, ones_matrix, rank, num_procs);   // batchSize x outputs
            mpi_compute_W2g(W2g, H, H_transpose, deltaOutput, rank, num_procs);    // hiddenSize x outputs
            mpi_compute_b2g(b2g, deltaOutput, rank, num_procs);    // 1 x outputs
            mpi_compute_He(He, deltaOutput, W2, W2_transpose, H, ones2_matrix, rank, num_procs);   // batchSize x hiddenSize
            mpi_compute_W1g(W1g, Xb, Xb_transpose, He, rank, num_procs);    // features x hiddenSize
            mpi_compute_b1g(b1g, He, rank, num_procs);    // 1 x hiddenSize

            // Update weights and biases
            mpi_update_weights(W1, W1g, eta, rank, num_procs);
            mpi_update_weights(W2, W2g, eta, rank, num_procs);
            mpi_update_weights(b1, b1g, eta, rank, num_procs);
            mpi_update_weights(b2, b2g, eta, rank, num_procs);

            // Free memory
            free_matrix(Xb);
            free_matrix(Yb);
        }
    }

    // Free memory
    free_matrix(H);
    free_matrix(Y_hat);
    free_matrix(E);
    free_matrix(deltaOutput);
    free_matrix(W2g);
    free_matrix(b2g);
    free_matrix(He);
    free_matrix(W1g);
    free_matrix(b1g);
    free_matrix(ones_matrix);
    free_matrix(ones2_matrix);
    free_matrix(H_transpose);
    free_matrix(W2_transpose);
    free_matrix(Xb_transpose);

    MLP_model model = {W1, W2, b1, b2};
    return model;
}

#endif
