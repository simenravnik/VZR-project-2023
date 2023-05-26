#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../read/read.h"
#include "../matrix/matrix.h"
#include "../models/mlp_model.h"

typedef struct TrainTestSplit {
    Matrix X_train;
    Matrix X_test;
    Matrix Y_train;
    Matrix Y_test;
} TrainTestSplit;

typedef struct Data {
    float* X;
    float* Y;
} Data;

TrainTestSplit prepare_dataset(const char* filepath, int classes);
TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize);
Data extract_data(DataFrame df);
TrainTestSplit split(float* X, float* Y, int samples, int features, int trainSize, int testSize);
Matrix normalize(Matrix X);
Matrix one_hot_encode(Matrix Y, int classes);
float accuracy_score(Matrix y_true, Matrix y_pred);
float predict(Matrix X_test, Matrix Y_test, MLP_model model);
void free_split(TrainTestSplit split);
void free_model(MLP_model model);

TrainTestSplit prepare_dataset(const char* filepath, int classes) {
    // Read data
    DataFrame df = read_csv(filepath);

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // TODO: For now we use the entire dataset for training
    int testSize = df.rows - trainSize;
    // int features = df.cols - 1;

    TrainTestSplit split = train_test_split(df, trainSize, testSize);

    // Normalize the input features
    split.X_train = normalize(split.X_train);
    split.X_test = normalize(split.X_test);

    // Perform one-hot encoding on labels
    split.Y_train = one_hot_encode(split.Y_train, classes);
    split.Y_test = one_hot_encode(split.Y_test, classes);

    // Free memory
    free(df.data);

    return split;
}

/*
 * Returns data splitted into train and test
*/
TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize) {

    // Split data into features and labels
    Data data = extract_data(df);

    // Split data into train and test
    TrainTestSplit tts = split(data.X, data.Y, df.rows, df.cols - 1, trainSize, testSize);

    return tts;
}


/*
 * Returns data splitted into features and labels
*/
Data extract_data(DataFrame df) {

    // Extract features
    float* X = (float*)malloc(sizeof(float) * df.rows * (df.cols - 1));
    for (int i = 0; i < df.rows; i++) {
        for (int j = 0; j < df.cols - 1; j++) {
            X[i * (df.cols - 1) + j] = df.data[i * df.cols + j];
        }
    }

    // Extract labels
    float* Y = (float*)malloc(sizeof(float) * df.rows);
    for (int i = 0; i < df.rows; i++) {
        Y[i] = df.data[i * df.cols + df.cols - 1];
    }

    Data data = {X, Y};
    return data;
}


/*
 * Returns data splitted into train and test
*/
TrainTestSplit split(float* X, float* Y, int samples, int features, int trainSize, int testSize) {

    Matrix X_train = allocate_matrix(trainSize, features);
    Matrix X_test = allocate_matrix(testSize, features);
    Matrix Y_train = allocate_matrix(trainSize, 1);
    Matrix Y_test = allocate_matrix(testSize, 1);

    // Assign values to X_train and X_test
    for (int i = 0; i < trainSize; i++) {
        for (int j = 0; j < features; j++) {
            X_train.data[i * features + j] = X[i * features + j];
        }
    }

    for (int i = 0; i < testSize; i++) {
        for (int j = 0; j < features; j++) {
            X_test.data[i * features + j] = X[(i + trainSize) * features + j];
        }
    }

    // Assign values to Y_train and Y_test
    for (int i = 0; i < trainSize; i++) {
        Y_train.data[i] = Y[i];
    }

    for (int i = 0; i < testSize; i++) {
        Y_test.data[i] = Y[i + trainSize];
    }

    TrainTestSplit tts = {X_train, X_test, Y_train, Y_test};
    return tts;
}

Matrix normalize(Matrix X) {

    Matrix X_norm = allocate_matrix(X.rows, X.cols);
    float* mean = (float*)malloc(sizeof(float) * X.cols);
    float* std = (float*)malloc(sizeof(float) * X.cols);

    // Calculate mean and std
    for (int i = 0; i < X.cols; i++) {
        float sum = 0;
        for (int j = 0; j < X.rows; j++) {
            sum += X.data[j * X.cols + i];
        }
        mean[i] = sum / X.rows;
    }

    for (int i = 0; i < X.cols; i++) {
        float sum = 0;
        for (int j = 0; j < X.rows; j++) {
            sum += pow(X.data[j * X.cols + i] - mean[i], 2);
        }
        std[i] = sqrt(sum / X.rows);
    }

    // Normalize
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            X_norm.data[i * X.cols + j] = (X.data[i * X.cols + j] - mean[j]) / std[j];
        }
    }

    return X_norm;
}

Matrix one_hot_encode(Matrix Y, int classes) {

    Matrix Y_encoded = allocate_matrix(Y.rows, classes);

    for (int i = 0; i < Y.rows; i++) {
        for (int j = 0; j < classes; j++) {
            if (Y.data[i] == j) {
                Y_encoded.data[i * classes + j] = 1;
            } else {
                Y_encoded.data[i * classes + j] = 0;
            }
        }
    }

    return Y_encoded;
}

float accuracy_score(Matrix y_true, Matrix y_pred) {

    Matrix y_true_argmax = argmax(y_true);
    Matrix y_pred_argmax = argmax(y_pred);

    float correct = 0;
    for (int i = 0; i < y_true.rows; i++) {
        if (y_true_argmax.data[i] == y_pred_argmax.data[i]) {
            correct++;
        }
    }

    return correct / y_true.rows;
}

float predict(Matrix X_test, Matrix Y_test, MLP_model model) {
    // Test the model
    Matrix H = matrix_tanh(add(dot(X_test, model.W1), model.b1));   // trainSize x hiddenSize
    Matrix Y_hat = matrix_tanh(add(dot(H, model.W2), model.b2));    // trainSize x classes

    // Calculate accuracy
    float accuracy = accuracy_score(Y_test, Y_hat);

    return accuracy;
}

void free_split(TrainTestSplit split) {
    free_matrix(split.X_train);
    free_matrix(split.X_test);
    free_matrix(split.Y_train);
    free_matrix(split.Y_test);
}

void free_model(MLP_model model) {
    free_matrix(model.W1);
    free_matrix(model.W2);
    free_matrix(model.b1);
    free_matrix(model.b2);
}

#endif
