#ifndef TRAIN_MLP_CUDA_NEW_H
#define TRAIN_MLP_CUDA_NEW_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../lib/matrix/matrix.h"
#include "../../lib/matrix/cuda_matrix.h"
#include "../../lib/matrix/cuda_matrix_new.h"
#include "../../lib/helpers/helpers.h"
#include "../../lib/helpers/helper_cuda.h"
#include "../../lib/models/mlp_model.h"

int calculate_block_size(int number) {
    // Array of target values
    int targets[] = {32, 64, 128, 256, 512, 1024};
    int numTargets = sizeof(targets) / sizeof(targets[0]);
    
    int closest = targets[0]; // Initialize closest to the first target
    
    for (int i = 1; i < numTargets; i++) {
        if (abs(targets[i] - number) < abs(closest - number)) {
            closest = targets[i]; // Update closest if a closer target is found
        }
    }

    return closest;
}

MLP_model train_mlp_cuda_new(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs) {

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

    int maxThreadsNeeded = MAX(MAX(MAX(MAX(batchSize * hiddenSize, batchSize * outputs), hiddenSize * outputs), batchSize * features), features * hiddenSize);

    int BS = calculate_block_size(maxThreadsNeeded); // BLOCKSIZE = Round up to the nearest multiple of 32
    int GS = (maxThreadsNeeded + BS - 1) / BS; // GRIDSIZE = 1

    printf("BlockSize = %d, GridSize = %d\n", BS, GS);

    if (maxThreadsNeeded > 1024) {
        printf("This model comes with a limitation of 1024 threads per block\n");
        printf("BlockSize too large, threads needed = %d\n", maxThreadsNeeded);
        exit(1);
    }

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: fix to encapsulate the last batch if samples % batchSize != 0
        for (int batch = 0; batch < samples; batch += batchSize) {

            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            Xb_dev = to_device(Xb);
            Yb_dev = to_device(Yb);

            train_on_gpu<<<GS, BS>>>(
                W1_dev.data, W2_dev.data, b1_dev.data, b2_dev.data, Xb_dev.data, Yb_dev.data,
                H_dev.data, Y_hat_dev.data, E_dev.data, deltaOutput_dev.data, W2g_dev.data, b2g_dev.data, He_dev.data, W1g_dev.data, b1g_dev.data,
                ones_dev.data, ones2_dev.data, H_transpose_dev.data, W2_transpose_dev.data, Xb_transpose_dev.data,
                batchSize, features, hiddenSize, outputs, eta, maxThreadsNeeded
            );

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
