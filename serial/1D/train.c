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
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} MLP_model;

MLP_model train_mlp(double* X, double* Y, int samples, int features, int outputs, int hiddenSize, double eta, int batchSize, int epochs) {

    // Initialize weights and biases
    double* W1 = random_matrix(features, hiddenSize);
    double* W2 = random_matrix(hiddenSize, outputs);
    double* b1 = random_matrix(1, hiddenSize);
    double* b2 = random_matrix(1, outputs);

    double *Xb, *Yb, *H, *Y_hat, *E, *deltaOutput, *W2g, *b2g, *He, *W1g, *b1g;

    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: fix to encapsulate the last batch if samples % batchSize != 0
        for (int batch = 0; batch < samples; batch += batchSize) {

            Xb = slice_matrix(X, batch, batch + batchSize, 0, features);   // batchSize x features
            Yb = slice_matrix(Y, batch, batch + batchSize, 0, outputs);    // batchSize x outputs

            // Forward pass
            H = matrix_tanh(add(dot(Xb, W1, batchSize, features, features, hiddenSize), b1, batchSize, hiddenSize), batchSize, hiddenSize);   // batchSize x hiddenSize
            Y_hat = matrix_tanh(add(dot(H, W2, batchSize, hiddenSize, hiddenSize, outputs), b2, batchSize, outputs), batchSize, outputs);    // batchSize x outputs

            // Backward pass
            E = subtract(Y_hat, Yb, batchSize, outputs);    // batchSize x outputs
            deltaOutput = hadamard(E, subtract(ones(batchSize, outputs), square(Y_hat, batchSize, outputs), batchSize, outputs), batchSize, outputs);   // batchSize x outputs
            W2g = dot(transpose(H, batchSize, hiddenSize), deltaOutput, hiddenSize, batchSize, batchSize, outputs);    // hiddenSize x outputs
            b2g = sum(deltaOutput, batchSize, outputs);    // 1 x outputs
            He = dot(deltaOutput, transpose(W2, hiddenSize, outputs), batchSize, outputs, outputs, hiddenSize);    // batchSize x hiddenSize
            He = hadamard(He, subtract(ones(batchSize, hiddenSize), square(H, batchSize, hiddenSize), batchSize, hiddenSize), batchSize, hiddenSize);    // batchSize x hiddenSize
            W1g = dot(transpose(Xb, batchSize, features), He, features, batchSize, batchSize, hiddenSize);    // features x hiddenSize
            b1g = sum(He, batchSize, hiddenSize);    // 1 x hiddenSize

            // Update weights and biases
            W1 = subtract(W1, scalar_multiply(W1g, eta, features, hiddenSize), features, hiddenSize);
            W2 = subtract(W2, scalar_multiply(W2g, eta, hiddenSize, outputs), hiddenSize, outputs);
            b1 = subtract(b1, scalar_multiply(b1g, eta, 1, hiddenSize), 1, hiddenSize);
            b2 = subtract(b2, scalar_multiply(b2g, eta, 1, outputs), 1, outputs);
        }
    }

    // Free memory
    free_matrix(Xb, batchSize);
    free_matrix(Yb, batchSize);
    free_matrix(H, batchSize);
    free_matrix(Y_hat, batchSize);
    free_matrix(E, batchSize);
    free_matrix(deltaOutput, batchSize);
    free_matrix(W2g, hiddenSize);
    free_matrix(b2g, 1);
    free_matrix(He, batchSize);
    free_matrix(W1g, features);
    free_matrix(b1g, 1);

    MLP_model model = {W1, W2, b1, b2};
    return model;
}

int main(int argc, char** argv) {

    // Read data
    DataFrame df = read_csv("../../data/iris.data");

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // TODO: For now we use the entire dataset for training
    int testSize = df.rows - trainSize;
    int features = df.cols - 1;

    TrainTestSplit split = train_test_split(df, trainSize, testSize);

    // Normalize the input features
    split.X_train = normalize(split.X_train, trainSize, features);
    split.X_test = normalize(split.X_test, testSize, features);

    // Perform one-hot encoding on labels
    int classes = 3;
    split.Y_train = one_hot_encode(split.Y_train, trainSize, classes);
    split.Y_test = one_hot_encode(split.Y_test, testSize, classes);

    int hiddenSize = 10;
    int batchSize = 10;
    int epochs = 1000;
    double eta = 0.01;

    // Train the model
    MLP_model model = train_mlp(split.X_train, split.Y_train, trainSize, features, classes, hiddenSize, eta, batchSize, epochs);

    // Test the model
    double* H = matrix_tanh(add(dot(split.X_train, model.W1, trainSize, features, features, hiddenSize), model.b1, trainSize, hiddenSize), trainSize, hiddenSize);   // trainSize x hiddenSize
    double* Y_hat = matrix_tanh(add(dot(H, model.W2, trainSize, hiddenSize, hiddenSize, classes), model.b2, trainSize, classes), trainSize, classes);    // trainSize x classes

    // Calculate accuracy
    double accuracy = accuracy_score(split.Y_train, Y_hat, trainSize, classes);
    printf("Accuracy: %f\n", accuracy);

    // Free memory
    free_matrix(df.data, df.rows);
    free_matrix(split.X_train, trainSize);
    free_matrix(split.Y_train, trainSize);
    free_matrix(split.X_test, testSize);
    free_matrix(split.Y_test, testSize);
    free_matrix(H, trainSize);
    free_matrix(Y_hat, trainSize);
    free_matrix(model.W1, features);
    free_matrix(model.W2, hiddenSize);
    free_matrix(model.b1, 1);
    free_matrix(model.b2, 1);
    
    return 0;
}