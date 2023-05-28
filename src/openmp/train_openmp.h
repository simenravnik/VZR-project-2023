#ifndef TRAIN_OPENMP_H
#define TRAIN_OPENMP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "../../lib/matrix/matrix.h"
#include "../../lib/matrix/matrix_openmp.h"
#include "../../lib/models/mlp_model.h"

MLP_model train_openmp(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs);

void omp_compute_H(Matrix H, Matrix Xb, Matrix W1, Matrix b1) {
    dot_omp(Xb, W1, H);
    add_omp(H, b1);
    matrix_tanh_omp(H);
}

void omp_compute_Y_hat(Matrix Y_hat, Matrix H, Matrix W2, Matrix b2) {
    dot_omp(H, W2, Y_hat);
    add_omp(Y_hat, b2);
    matrix_tanh_omp(Y_hat);
}

void omp_compute_E(Matrix E, Matrix Y_hat, Matrix Yb) {
    subtract_omp(Y_hat, Yb, E);
}

void omp_compute_delta_output(Matrix deltaOutput, Matrix E, Matrix Y_hat, Matrix ones_matrix) {
    square_omp(Y_hat);
    subtract_omp(ones_matrix, Y_hat, Y_hat);
    hadamard_omp(E, Y_hat, deltaOutput);
}

void omp_compute_W2g(Matrix W2g, Matrix H, Matrix H_tranpose, Matrix deltaOutput) {
    transpose_omp(H, H_tranpose);
    dot_omp(H_tranpose, deltaOutput, W2g);
}

void omp_compute_b2g(Matrix b2g, Matrix deltaOutput) {
    sum_omp(deltaOutput, b2g);
}

void omp_compute_He(Matrix He, Matrix deltaOutput, Matrix W2, Matrix W2_transpose, Matrix H, Matrix ones2_matrix) {
    transpose_omp(W2, W2_transpose);
    dot_omp(deltaOutput, W2_transpose, He);
    square_omp(H);
    subtract_omp(ones2_matrix, H, H);
    hadamard_omp(He, H, He);
}

void omp_compute_W1g(Matrix W1g, Matrix Xb, Matrix Xb_transpose, Matrix He) {
    transpose_omp(Xb, Xb_transpose);
    dot_omp(Xb_transpose, He, W1g);
}

void omp_compute_b1g(Matrix b1g, Matrix He) {
    sum_omp(He, b1g);
}

void omp_update_weights(Matrix m, Matrix g, float eta) {
    scalar_multiply_omp(g, eta);
    subtract_omp(m, g, m);
}

MLP_model train_openmp(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs) {

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
            omp_compute_H(H, Xb, W1, b1);   // batchSize x hiddenSize
            omp_compute_Y_hat(Y_hat, H, W2, b2);    // batchSize x outputs

            // Backward pass
            omp_compute_E(E, Y_hat, Yb);    // batchSize x outputs
            omp_compute_delta_output(deltaOutput, E, Y_hat, ones_matrix);   // batchSize x outputs
            omp_compute_W2g(W2g, H, H_transpose, deltaOutput);    // hiddenSize x outputs
            omp_compute_b2g(b2g, deltaOutput);    // 1 x outputs
            omp_compute_He(He, deltaOutput, W2, W2_transpose, H, ones2_matrix);   // batchSize x hiddenSize
            omp_compute_W1g(W1g, Xb, Xb_transpose, He);    // features x hiddenSize
            omp_compute_b1g(b1g, He);    // 1 x hiddenSize

            // Update weights and biases
            omp_update_weights(W1, W1g, eta);
            omp_update_weights(W2, W2g, eta);
            omp_update_weights(b1, b1g, eta);
            omp_update_weights(b2, b2g, eta);

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
