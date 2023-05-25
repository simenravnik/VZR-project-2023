/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lib/read/read.h"
#include "lib/matrix/matrix.h"
#include "lib/helpers/helpers.h"
#include "lib/matrix/cuda_matrix.h"
#include "lib/models/mlp_model.h"
#include "lib/time/cuda_timer.h"

#include "src/serial/train_mlp_serial.h"
#include "src/cuda/train_mlp_cuda.h"
#include "src/cuda/train_mlp_cuda_new.h"

void predict(Matrix X_test, Matrix Y_test, MLP_model model) {
    // Test the model
    Matrix H = matrix_tanh(add(dot(X_test, model.W1), model.b1));   // trainSize x hiddenSize
    Matrix Y_hat = matrix_tanh(add(dot(H, model.W2), model.b2));    // trainSize x classes

    // Calculate accuracy
    float accuracy = accuracy_score(Y_test, Y_hat);
    printf("Accuracy: %f\n\n", accuracy);
}

int main(int argc, char** argv) {

    // Read data
    DataFrame df = read_csv("data/winequality-red.csv");

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // TODO: For now we use the entire dataset for training
    int testSize = df.rows - trainSize;
    // int features = df.cols - 1;

    TrainTestSplit split = train_test_split(df, trainSize, testSize);

    // Normalize the input features
    split.X_train = normalize(split.X_train);
    split.X_test = normalize(split.X_test);

    // Perform one-hot encoding on labels
    int classes = 10;
    split.Y_train = one_hot_encode(split.Y_train, classes);
    split.Y_test = one_hot_encode(split.Y_test, classes);

    int hiddenSize = 50;
    int batchSize = 10;
    int epochs = 100;
    float eta = 0.01;

    MLP_model model;
    
    // Start timer
    cudaEvent_t start, stop;
    cuda_start_timer(&start, &stop);

    // Train the model
    if (strcmp(argv[1], "serial")) {
        model = train_mlp_serial(split.X_train, split.Y_train, hiddenSize, eta, batchSize, epochs);
    } else if (strcmp(argv[1], "cuda")) {
        model = train_mlp_cuda(split.X_train, split.Y_train, hiddenSize, eta, batchSize, epochs);
    } else if (strcmp(argv[1], "cuda_new")) {
        model = train_mlp_cuda_new(split.X_train, split.Y_train, hiddenSize, eta, batchSize, epochs);
    } else {
        printf("Invalid argument. Please use either 'serial' or 'cuda'.\n");
        return 1;
    }
    
    // Stop timer
    printf("Time: %0.3f milliseconds \n", cuda_stop_timer(&start, &stop));

    // Test the model
    predict(split.X_train, split.Y_train, model);

    // Free memory
    free(df.data);
    free_matrix(split.X_train);
    free_matrix(split.Y_train);
    free_matrix(split.X_test);
    free_matrix(split.Y_test);
    free_matrix(model.W1);
    free_matrix(model.W2);
    free_matrix(model.b1);
    free_matrix(model.b2);
    
    return 0;
}