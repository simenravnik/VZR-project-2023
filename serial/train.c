#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "train_test_split.h"

typedef struct MLP_model {
    double** W1;
    double** W2;
    double** b1;
    double** b2;
} MLP_model;

MLP_model train_mlp(double** X, double** Y, int samples, int features, int outputs, int hiddenSize, double eta, int batchSize, int epochs) {

    // Initialize weights and biases
    double** W1 = random_matrix(features, hiddenSize);
    double** W2 = random_matrix(hiddenSize, outputs);
    double** b1 = random_matrix(1, hiddenSize);
    double** b2 = random_matrix(1, outputs);

    print_matrix(W1, features, hiddenSize);
    print_matrix(W2, hiddenSize, outputs);
    print_matrix(b1, 1, hiddenSize);
    print_matrix(b2, 1, outputs);

    MLP_model model = {W1, W2, b1, b2};
    return model;
}

int main(int argc, char** argv) {

    // Read data
    DataFrame df = read_csv("../data/iris.data");

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // For now we use the entire dataset for training
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

    print_matrix(split.X_train, trainSize, features);
    print_matrix(split.Y_train, trainSize, classes);
    print_matrix(split.X_test, testSize, features);
    print_matrix(split.Y_test, testSize, classes);

    int hiddenSize = 3;
    int batchSize = 1;
    int epochs = 100;
    double eta = 0.01;

    // Train the model
    MLP_model model = train_mlp(split.X_train, split.Y_test, trainSize, features, classes, hiddenSize, eta, batchSize, epochs);
    
    return 0;
}