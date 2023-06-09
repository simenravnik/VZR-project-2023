#ifndef TRAIN_CUDA_H
#define TRAIN_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../lib/matrix/matrix.h"
#include "../../lib/matrix/matrix_cuda.h"
#include "../../lib/helpers/helpers.h"
#include "../../lib/helpers/helper_cuda.h"
#include "../../lib/models/mlp_model.h"
#include "../../parameters.h"

void compute_H(Matrix H_dev, Matrix Xb_dev, Matrix W1_dev, Matrix b1_dev);
void compute_Y_hat(Matrix Y_hat_dev, Matrix H_dev, Matrix W2_dev, Matrix b2_dev);
void compute_E(Matrix E, Matrix Y_hat, Matrix Yb);
void compute_delta_output(Matrix deltaOutput_dev, Matrix E_dev, Matrix ones_dev, Matrix Y_hat_dev);
void compute_w2g(Matrix W2g_dev, Matrix H_dev, Matrix H_transpose_dev, Matrix deltaOutput_dev);
void compute_b2g(Matrix b2g_dev, Matrix deltaOutput_dev);
void compute_He(Matrix He_dev, Matrix deltaOutput_dev, Matrix W2_dev, Matrix W2_transpose_dev, Matrix H_dev, Matrix ones2_dev);
void compute_W1g(Matrix W1g_dev, Matrix Xb_dev, Matrix Xb_transpose_dev, Matrix He_dev);
void compute_b1g(Matrix b1g_dev, Matrix He_dev);
void update_weights(Matrix m, Matrix g, float eta);

MLP_model train_cuda(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs);

void compute_H(Matrix H_dev, Matrix Xb_dev, Matrix W1_dev, Matrix b1_dev) {

    int batchSize = Xb_dev.rows;
    int features = Xb_dev.cols;
    int hiddenSize = W1_dev.cols;

    int gridSize = (batchSize * hiddenSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_dot<<<gridSize, CUDA_BLOCK_SIZE>>>(Xb_dev.data, W1_dev.data, H_dev.data, batchSize, features, hiddenSize);
    device_add<<<gridSize, CUDA_BLOCK_SIZE>>>(H_dev.data, b1_dev.data, H_dev.data, batchSize, hiddenSize);
    device_matrix_tanh<<<gridSize, CUDA_BLOCK_SIZE>>>(H_dev.data, H_dev.data, batchSize, hiddenSize);
}

void compute_Y_hat(Matrix Y_hat_dev, Matrix H_dev, Matrix W2_dev, Matrix b2_dev) {

    int batchSize = H_dev.rows;
    int hiddenSize = H_dev.cols;
    int outputs = W2_dev.cols;

    int gridSize = (batchSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_dot<<<gridSize, CUDA_BLOCK_SIZE>>>(H_dev.data, W2_dev.data, Y_hat_dev.data, batchSize, hiddenSize, outputs);
    device_add<<<gridSize, CUDA_BLOCK_SIZE>>>(Y_hat_dev.data, b2_dev.data, Y_hat_dev.data, batchSize, outputs);
    device_matrix_tanh<<<gridSize, CUDA_BLOCK_SIZE>>>(Y_hat_dev.data, Y_hat_dev.data, batchSize, outputs);
}

void compute_E(Matrix E, Matrix Y_hat, Matrix Yb) {
    
    int batchSize = Y_hat.rows;
    int outputs = Y_hat.cols;

    int gridSize = (batchSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_subtract<<<gridSize, CUDA_BLOCK_SIZE>>>(Y_hat.data, Yb.data, E.data, batchSize, outputs);
}

void compute_delta_output(Matrix deltaOutput_dev, Matrix E_dev, Matrix ones_dev, Matrix Y_hat_dev) {
    
    int batchSize = E_dev.rows;
    int outputs = E_dev.cols;

    int gridSize = (batchSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_square<<<gridSize, CUDA_BLOCK_SIZE>>>(Y_hat_dev.data, Y_hat_dev.data, batchSize, outputs);
    device_subtract<<<gridSize, CUDA_BLOCK_SIZE>>>(ones_dev.data, Y_hat_dev.data, deltaOutput_dev.data, batchSize, outputs);
    device_hadamard<<<gridSize, CUDA_BLOCK_SIZE>>>(E_dev.data, deltaOutput_dev.data, deltaOutput_dev.data, batchSize, outputs);
}

void compute_w2g(Matrix W2g_dev, Matrix H_dev, Matrix H_transpose_dev, Matrix deltaOutput_dev) {

    int batchSize = H_dev.rows;
    int hiddenSize = H_dev.cols;
    int outputs = deltaOutput_dev.cols;

    int gridSizeTranspose = (batchSize * hiddenSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridSizeDot = (hiddenSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_transpose<<<gridSizeTranspose, CUDA_BLOCK_SIZE>>>(H_dev.data, H_transpose_dev.data, batchSize, hiddenSize);
    device_dot<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(H_transpose_dev.data, deltaOutput_dev.data, W2g_dev.data, hiddenSize, batchSize, outputs);
}

void compute_b2g(Matrix b2g_dev, Matrix deltaOutput_dev) {

    int batchSize = deltaOutput_dev.rows;
    int outputs = deltaOutput_dev.cols;

    int gridSize = (batchSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_sum<<<gridSize, CUDA_BLOCK_SIZE>>>(deltaOutput_dev.data, b2g_dev.data, batchSize, outputs);
}

void compute_He(Matrix He_dev, Matrix deltaOutput_dev, Matrix W2_dev, Matrix W2_transpose_dev, Matrix H_dev, Matrix ones2_dev) {

    int batchSize = deltaOutput_dev.rows;
    int outputs = deltaOutput_dev.cols;
    int hiddenSize = W2_dev.rows;

    int gridSizeTranspose = (hiddenSize * outputs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridSizeDot = (batchSize * hiddenSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_transpose<<<gridSizeTranspose, CUDA_BLOCK_SIZE>>>(W2_dev.data, W2_transpose_dev.data, hiddenSize, outputs);
    device_dot<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(deltaOutput_dev.data, W2_transpose_dev.data, He_dev.data, batchSize, outputs, hiddenSize);

    device_square<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(H_dev.data, H_dev.data, batchSize, hiddenSize);
    device_subtract<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(ones2_dev.data, H_dev.data, H_dev.data, batchSize, hiddenSize);
    device_hadamard<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(He_dev.data, H_dev.data, He_dev.data, batchSize, hiddenSize);
}

void compute_W1g(Matrix W1g_dev, Matrix Xb_dev, Matrix Xb_transpose_dev, Matrix He_dev) {

    int batchSize = Xb_dev.rows;
    int features = Xb_dev.cols;
    int hiddenSize = He_dev.cols;

    int gridSizeTranspose = (batchSize * features + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridSizeDot = (features * hiddenSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_transpose<<<gridSizeTranspose, CUDA_BLOCK_SIZE>>>(Xb_dev.data, Xb_transpose_dev.data, batchSize, features);
    device_dot<<<gridSizeDot, CUDA_BLOCK_SIZE>>>(Xb_transpose_dev.data, He_dev.data, W1g_dev.data, features, batchSize, hiddenSize);
}

void compute_b1g(Matrix b1g_dev, Matrix He_dev) {

    int batchSize = He_dev.rows;
    int hiddenSize = He_dev.cols;

    int gridSize = (hiddenSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_sum<<<gridSize, CUDA_BLOCK_SIZE>>>(He_dev.data, b1g_dev.data, batchSize, hiddenSize);
}

void update_weights(Matrix m, Matrix g, float eta) {

    int rows = m.rows;
    int cols = m.cols;

    int gridSize = (rows * cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    device_scalar_multiply<<<gridSize, CUDA_BLOCK_SIZE>>>(g.data, g.data, eta, rows, cols);
    device_subtract<<<gridSize, CUDA_BLOCK_SIZE>>>(m.data, g.data, m.data, rows, cols);
}

MLP_model train_cuda(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs) {

    int samples = X.rows;
    int features = X.cols;
    int outputs = Y.cols;

    // Initialize weights and biases
    Matrix W1 = random_matrix(features, hiddenSize);
    Matrix W2 = random_matrix(hiddenSize, outputs);
    Matrix b1 = random_matrix(1, hiddenSize);
    Matrix b2 = random_matrix(1, outputs);

    // Push weights and biases to the device
    Matrix W1_dev = to_device(W1);
    Matrix W2_dev = to_device(W2);
    Matrix b1_dev = to_device(b1);
    Matrix b2_dev = to_device(b2);

    // Initialize matries for the forward and backward pass
    Matrix H_dev = create_on_device(batchSize, hiddenSize);
    Matrix Y_hat_dev = create_on_device(batchSize, outputs);
    Matrix E_dev = create_on_device(batchSize, outputs);
    Matrix deltaOutput_dev = create_on_device(batchSize, outputs);
    Matrix W2g_dev = create_on_device(hiddenSize, outputs);
    Matrix b2g_dev = create_on_device(1, outputs);
    Matrix He_dev = create_on_device(batchSize, hiddenSize);
    Matrix W1g_dev = create_on_device(features, hiddenSize);
    Matrix b1g_dev = create_on_device(1, hiddenSize);
    Matrix ones_dev = to_device(ones(batchSize, outputs));
    Matrix ones2_dev = to_device(ones(batchSize, hiddenSize));
    Matrix H_transpose_dev = create_on_device(hiddenSize, batchSize);  // Helper matrix for the transpose of H
    Matrix W2_transpose_dev = create_on_device(outputs, hiddenSize);    // Helper matrix for the transpose of W2
    Matrix Xb_transpose_dev = create_on_device(features, batchSize);    // Helper matrix for the transpose of Xb

    Matrix Xb, Yb;
    Matrix Xb_dev, Yb_dev;

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: fix to encapsulate the last batch if samples % batchSize != 0
        for (int batch = 0; batch < samples; batch += batchSize) {

            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            Xb_dev = to_device(Xb);
            Yb_dev = to_device(Yb);

            // Forward pass
            compute_H(H_dev, Xb_dev, W1_dev, b1_dev);   // batchSize x hiddenSize 
            compute_Y_hat(Y_hat_dev, H_dev, W2_dev, b2_dev);    // batchSize x outputs

            // Backward pass
            compute_E(E_dev, Y_hat_dev, Yb_dev);    // batchSize x outputs
            compute_delta_output(deltaOutput_dev, E_dev, ones_dev, Y_hat_dev);  // batchSize x outputs
            compute_w2g(W2g_dev, H_dev, H_transpose_dev, deltaOutput_dev);    // hiddenSize x outputs
            compute_b2g(b2g_dev, deltaOutput_dev);    // 1 x outputs
            compute_He(He_dev, deltaOutput_dev, W2_dev, W2_transpose_dev, H_dev, ones2_dev);    // batchSize x hiddenSize
            compute_W1g(W1g_dev, Xb_dev, Xb_transpose_dev, He_dev);    // features x hiddenSize
            compute_b1g(b1g_dev, He_dev);    // 1 x hiddenSize

            // Update weights and biases
            update_weights(W1_dev, W1g_dev, eta);
            update_weights(W2_dev, W2g_dev, eta);
            update_weights(b1_dev, b1g_dev, eta);
            update_weights(b2_dev, b2g_dev, eta);

            // Free memory
            free_matrix(Xb);
            free_matrix(Yb);
            free_device_matrix(Xb_dev);
            free_device_matrix(Yb_dev);
        }
    }

    // Push computed weights and biases to the host
    W1 = to_host(W1_dev);
    W2 = to_host(W2_dev);
    b1 = to_host(b1_dev);
    b2 = to_host(b2_dev);

    // Free memory
    free_device_matrix(H_dev);
    free_device_matrix(Y_hat_dev);
    free_device_matrix(E_dev);
    free_device_matrix(deltaOutput_dev);
    free_device_matrix(W2g_dev);
    free_device_matrix(b2g_dev);
    free_device_matrix(He_dev);
    free_device_matrix(W1g_dev);
    free_device_matrix(b1g_dev);
    free_device_matrix(ones_dev);
    free_device_matrix(ones2_dev);
    free_device_matrix(H_transpose_dev);
    free_device_matrix(W2_transpose_dev);
    free_device_matrix(Xb_transpose_dev);
    free_device_matrix(W1_dev);
    free_device_matrix(W2_dev);
    free_device_matrix(b1_dev);
    free_device_matrix(b2_dev);

    MLP_model model = {W1, W2, b1, b2};
    return model;
}

#endif
