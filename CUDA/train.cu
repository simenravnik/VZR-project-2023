/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lib/read.h"
#include "lib/matrix.h"
#include "lib/helpers.h"
#include "lib/cuda_matrix.h"

typedef struct MLP_model {
    Matrix W1;
    Matrix W2;
    Matrix b1;
    Matrix b2;
} MLP_model;

void compute_H(Matrix H_dev, Matrix Xb_dev, Matrix W1_dev, Matrix b1_dev) {

    int batchSize = Xb_dev.rows;
    int features = Xb_dev.cols;
    int hiddenSize = W1_dev.cols;

    // Define the block and grid size
    int blockSize = 32;
    int gridSize = (batchSize * hiddenSize + blockSize - 1) / blockSize;

    // Compute the hidden layer
    device_dot<<<gridSize, blockSize>>>(Xb_dev.data, W1_dev.data, H_dev.data, batchSize, features, hiddenSize);
    device_add<<<gridSize, blockSize>>>(H_dev.data, b1_dev.data, H_dev.data, batchSize, hiddenSize, b1_dev.rows, b1_dev.cols);
    device_matrix_tanh<<<gridSize, blockSize>>>(H_dev.data, H_dev.data, batchSize, hiddenSize);
}

void compute_Y_hat(Matrix Y_hat_dev, Matrix H_dev, Matrix W2_dev, Matrix b2_dev) {

    int batchSize = H_dev.rows;
    int hiddenSize = H_dev.cols;
    int outputs = W2_dev.cols;

    // Define the block and grid size
    int blockSize = 32;
    int gridSize = (batchSize * outputs + blockSize - 1) / blockSize;

    // Compute the output layer
    device_dot<<<gridSize, blockSize>>>(H_dev.data, W2_dev.data, Y_hat_dev.data, batchSize, hiddenSize, outputs);
    device_add<<<gridSize, blockSize>>>(Y_hat_dev.data, b2_dev.data, Y_hat_dev.data, batchSize, outputs, b2_dev.rows, b2_dev.cols);
    device_matrix_tanh<<<gridSize, blockSize>>>(Y_hat_dev.data, Y_hat_dev.data, batchSize, outputs);
}

void compute_E(Matrix E, Matrix Y_hat, Matrix Yb) {
    
    int batchSize = Y_hat.rows;
    int outputs = Y_hat.cols;

    // Define the block and grid size
    int blockSize = 32;
    int gridSize = (batchSize * outputs + blockSize - 1) / blockSize;

    // Compute the error
    device_subtract<<<gridSize, blockSize>>>(Y_hat.data, Yb.data, E.data, batchSize, outputs, Yb.rows, Yb.cols);
}

void compute_delta_output(Matrix deltaOutput_dev, Matrix E_dev, Matrix ones_dev, Matrix Y_hat_dev) {
    
        int batchSize = E_dev.rows;
        int outputs = E_dev.cols;
    
        // Define the block and grid size
        int blockSize = 32;
        int gridSize = (batchSize * outputs + blockSize - 1) / blockSize;
    
        device_square<<<gridSize, blockSize>>>(Y_hat_dev.data, Y_hat_dev.data, batchSize, outputs);
        device_subtract<<<gridSize, blockSize>>>(ones_dev.data, Y_hat_dev.data, deltaOutput_dev.data, batchSize, outputs, ones_dev.rows, ones_dev.cols);
        device_hadamard<<<gridSize, blockSize>>>(E_dev.data, deltaOutput_dev.data, deltaOutput_dev.data, batchSize, outputs, batchSize, outputs);
}

void compute_w2g(Matrix W2g_dev, Matrix H_dev, Matrix deltaOutput_dev) {

    // Define the block and grid size
    int blockSize = 32;
    int gridSizeTranspose = (H_dev.rows * H_dev.cols + blockSize - 1) / blockSize;
    int gridSizeDot = (H_dev.cols * deltaOutput_dev.cols + blockSize - 1) / blockSize;

    device_transpose<<<gridSizeTranspose, blockSize>>>(H_dev.data, W2g_dev.data, H_dev.rows, H_dev.cols);
    device_dot<<<gridSizeDot, blockSize>>>(W2g_dev.data, deltaOutput_dev.data, W2g_dev.data, H_dev.cols, H_dev.rows, deltaOutput_dev.cols);
}

void compute_b2g(Matrix b2g_dev, Matrix deltaOutput_dev) {

    int batchSize = deltaOutput_dev.rows;
    int outputs = deltaOutput_dev.cols;

    // Define the block and grid size
    int blockSize = 32;
    int gridSize = (batchSize * outputs + blockSize - 1) / blockSize;

    device_sum<<<gridSize, blockSize>>>(deltaOutput_dev.data, b2g_dev.data, batchSize, outputs);
}

void compute_He(Matrix He_dev, Matrix deltaOutput_dev, Matrix W2_dev, Matrix H_dev, Matrix ones2_dev) {

    int batchSize = deltaOutput_dev.rows;
    int outputs = deltaOutput_dev.cols;
    int hiddenSize = W2_dev.rows;

    // Define the block and grid size
    int blockSize = 32;
    int gridSizeTranspose = (hiddenSize * outputs + blockSize - 1) / blockSize;
    int gridSizeDot = (batchSize * hiddenSize + blockSize - 1) / blockSize;

    device_transpose<<<gridSizeTranspose, blockSize>>>(W2_dev.data, He_dev.data, hiddenSize, outputs);
    device_dot<<<gridSizeDot, blockSize>>>(deltaOutput_dev.data, He_dev.data, He_dev.data, batchSize, outputs, hiddenSize);

    device_square<<<gridSizeDot, blockSize>>>(H_dev.data, H_dev.data, batchSize, hiddenSize);
    device_subtract<<<gridSizeDot, blockSize>>>(ones2_dev.data, H_dev.data, H_dev.data, batchSize, hiddenSize, ones2_dev.rows, ones2_dev.cols);
    device_hadamard<<<gridSizeDot, blockSize>>>(He_dev.data, H_dev.data, He_dev.data, batchSize, hiddenSize, batchSize, hiddenSize);
}

void compute_W1g(Matrix W1g_dev, Matrix Xb_dev, Matrix Xb_transpose_dev, Matrix He_dev) {

    int batchSize = Xb_dev.rows;
    int features = Xb_dev.cols;
    int hiddenSize = He_dev.cols;

    // Define the block and grid size
    int blockSize = 32;
    int gridSizeTranspose = (batchSize * features + blockSize - 1) / blockSize;
    int gridSizeDot = (features * hiddenSize + blockSize - 1) / blockSize;

    device_transpose<<<gridSizeTranspose, blockSize>>>(Xb_dev.data, Xb_transpose_dev.data, batchSize, features);
    device_dot<<<gridSizeDot, blockSize>>>(Xb_transpose_dev.data, He_dev.data, W1g_dev.data, features, batchSize, hiddenSize);
}

MLP_model train_mlp(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs) {

    int samples = X.rows;
    int features = X.cols;
    int outputs = Y.cols;

    // Initialize weights and biases
    Matrix W1 = random_matrix(features, hiddenSize);
    Matrix W2 = random_matrix(hiddenSize, outputs);
    Matrix b1 = random_matrix(1, hiddenSize);
    Matrix b2 = random_matrix(1, outputs);

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
    Matrix Xb_transpose_dev = create_on_device(features, batchSize);    // Helper matrix for the transpose of Xb

    Matrix Xb, Yb, H, Y_hat, E, deltaOutput, W2g, b2g, He, W1g, b1g;

    Matrix Xb_dev, Yb_dev;

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: fix to encapsulate the last batch if samples % batchSize != 0
        for (int batch = 0; batch < samples; batch += batchSize) {

            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            Xb_dev = to_device(Xb);
            Yb_dev = to_device(Yb);

            // Push weights and biases to the device
            Matrix W1_dev = to_device(W1);
            Matrix W2_dev = to_device(W2);
            Matrix b1_dev = to_device(b1);
            Matrix b2_dev = to_device(b2);

            // Forward pass
            compute_H(H_dev, Xb_dev, W1_dev, b1_dev);   // batchSize x hiddenSize 
            compute_Y_hat(Y_hat_dev, H_dev, W2_dev, b2_dev);    // batchSize x outputs

            // Backward pass
            compute_E(E_dev, Y_hat_dev, Yb_dev);    // batchSize x outputs
            compute_delta_output(deltaOutput_dev, E_dev, ones_dev, Y_hat_dev);  // batchSize x outputs
            compute_w2g(W2g_dev, H_dev, deltaOutput_dev);    // hiddenSize x outputs
            compute_b2g(b2g_dev, deltaOutput_dev);    // 1 x outputs
            compute_He(He_dev, deltaOutput_dev, W2_dev, H_dev, ones2_dev);    // batchSize x hiddenSize
            compute_W1g(W1g_dev, Xb_dev, Xb_transpose_dev, He_dev);    // features x hiddenSize

            H = to_host(H_dev);
            Y_hat = to_host(Y_hat_dev);
            E = to_host(E_dev);
            deltaOutput = to_host(deltaOutput_dev);
            W2g = to_host(W2g_dev);
            b2g = to_host(b2g_dev);
            He = to_host(He_dev);
            W1g = to_host(W1g_dev);

            print_matrix(W1g);
            
            b1g = sum(He);    // 1 x hiddenSize

            // Update weights and biases
            W1 = subtract(W1, scalar_multiply(W1g, eta));
            W2 = subtract(W2, scalar_multiply(W2g, eta));
            b1 = subtract(b1, scalar_multiply(b1g, eta));
            b2 = subtract(b2, scalar_multiply(b2g, eta));

            if (batch == batchSize)
                break;
        }
        break;
    }

    // Free memory
    free_matrix(Xb);
    free_matrix(Yb);
    free_matrix(H);
    free_matrix(Y_hat);
    free_matrix(E);
    free_matrix(deltaOutput);
    free_matrix(W2g);
    free_matrix(b2g);
    free_matrix(He);
    free_matrix(W1g);
    free_matrix(b1g);

    MLP_model model = {W1, W2, b1, b2};
    return model;
}

int main(int argc, char** argv) {

    // Read data
    DataFrame df = read_csv("../data/iris.data");

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // TODO: For now we use the entire dataset for training
    int testSize = df.rows - trainSize;
    int features = df.cols - 1;

    TrainTestSplit split = train_test_split(df, trainSize, testSize);

    // Normalize the input features
    split.X_train = normalize(split.X_train);
    split.X_test = normalize(split.X_test);

    // Perform one-hot encoding on labels
    int classes = 3;
    split.Y_train = one_hot_encode(split.Y_train, classes);
    split.Y_test = one_hot_encode(split.Y_test, classes);

    int hiddenSize = 10;
    int batchSize = 10;
    int epochs = 1000;
    float eta = 0.01;

    // Train the model
    MLP_model model = train_mlp(split.X_train, split.Y_train, hiddenSize, eta, batchSize, epochs);

    // Test the model
    Matrix H = matrix_tanh(add(dot(split.X_train, model.W1), model.b1));   // trainSize x hiddenSize
    Matrix Y_hat = matrix_tanh(add(dot(H, model.W2), model.b2));    // trainSize x classes

    // Calculate accuracy
    float accuracy = accuracy_score(split.Y_train, Y_hat);
    printf("Accuracy: %f\n", accuracy);

    // Free memory
    free(df.data);
    free_matrix(split.X_train);
    free_matrix(split.Y_train);
    free_matrix(split.X_test);
    free_matrix(split.Y_test);
    free_matrix(H);
    free_matrix(Y_hat);
    free_matrix(model.W1);
    free_matrix(model.W2);
    free_matrix(model.b1);
    free_matrix(model.b2);
    
    return 0;
}