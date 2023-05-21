/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lib/read.h"
#include "lib/matrix.h"
#include "lib/helpers.h"

typedef struct MLP_model {
    Matrix W1;
    Matrix W2;
    Matrix b1;
    Matrix b2;
} MLP_model;

MLP_model train_mlp(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs) {

    int samples = X.rows;
    int features = X.cols;
    int outputs = Y.cols;

    // Initialize weights and biases
    Matrix W1 = random_matrix(features, hiddenSize);
    Matrix W2 = random_matrix(hiddenSize, outputs);
    Matrix b1 = random_matrix(1, hiddenSize);
    Matrix b2 = random_matrix(1, outputs);

    Matrix Xb, Yb, H, Y_hat, E, deltaOutput, W2g, b2g, He, W1g, b1g;

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: fix to encapsulate the last batch if samples % batchSize != 0
        for (int batch = 0; batch < samples; batch += batchSize) {

            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            // Forward pass
            H = matrix_tanh(add(dot(Xb, W1), b1));   // batchSize x hiddenSize
            Y_hat = matrix_tanh(add(dot(H, W2), b2));    // batchSize x outputs

            // Backward pass
            E = subtract(Y_hat, Yb);    // batchSize x outputs
            deltaOutput = hadamard(E, subtract(ones(batchSize, outputs), square(Y_hat)));   // batchSize x outputs
            W2g = dot(transpose(H), deltaOutput);    // hiddenSize x outputs
            b2g = sum(deltaOutput);    // 1 x outputs
            He = dot(deltaOutput, transpose(W2));    // batchSize x hiddenSize
            He = hadamard(He, subtract(ones(batchSize, hiddenSize), square(H)));    // batchSize x hiddenSize
            W1g = dot(transpose(Xb), He);    // features x hiddenSize
            b1g = sum(He);    // 1 x hiddenSize

            // Update weights and biases
            W1 = subtract(W1, scalar_multiply(W1g, eta));
            W2 = subtract(W2, scalar_multiply(W2g, eta));
            b1 = subtract(b1, scalar_multiply(b1g, eta));
            b2 = subtract(b2, scalar_multiply(b2g, eta));
        }
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
    DataFrame df = read_csv("../../../data/iris.data");

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