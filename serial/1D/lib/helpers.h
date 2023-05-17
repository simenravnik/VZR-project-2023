#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "read.h"
#include "matrix.h"

typedef struct TrainTestSplit {
    double* X_train;
    double* X_test;
    double* Y_train;
    double* Y_test;
    int trainSize;
    int testSize;
    int features;
} TrainTestSplit;

typedef struct Data {
    double* X;
    double* Y;
} Data;

TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize);
Data extract_data(DataFrame df);
TrainTestSplit split(double* X, double* Y, int samples, int features, int trainSize, int testSize);
double* normalize(double* X, int samples, int features);
double* one_hot_encode(double* Y, int samples, int classes);
double accuracy_score(double* y_true, double* y_pred, int rows, int cols);

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
    double* X = (double*)malloc(sizeof(double) * df.rows * (df.cols - 1));
    for (int i = 0; i < df.rows; i++) {
        for (int j = 0; j < df.cols - 1; j++) {
            X[i * (df.cols - 1) + j] = df.data[i * df.cols + j];
        }
    }

    // Extract labels
    double* Y = (double*)malloc(sizeof(double) * df.rows);
    for (int i = 0; i < df.rows; i++) {
        Y[i] = df.data[i * df.cols + df.cols - 1];
    }

    Data data = {X, Y};
    return data;
}


/*
 * Returns data splitted into train and test
*/
TrainTestSplit split(double* X, double* Y, int samples, int features, int trainSize, int testSize) {

    double* X_train = (double*)malloc(sizeof(double) * trainSize * features);
    double* X_test = (double*)malloc(sizeof(double) * testSize * features);
    double* Y_train = (double*)malloc(sizeof(double) * trainSize);
    double* Y_test = (double*)malloc(sizeof(double) * testSize);

    // Assign values to X_train and X_test
    for (int i = 0; i < trainSize; i++) {
        for (int j = 0; j < features; j++) {
            X_train[i * features + j] = X[i * features + j];
        }
    }

    for (int i = 0; i < testSize; i++) {
        for (int j = 0; j < features; j++) {
            X_test[i * features + j] = X[(i + trainSize) * features + j];
        }
    }

    // Assign values to Y_train and Y_test
    for (int i = 0; i < trainSize; i++) {
        Y_train[i] = Y[i];
    }

    for (int i = 0; i < testSize; i++) {
        Y_test[i] = Y[i + trainSize];
    }

    TrainTestSplit tts = {X_train, X_test, Y_train, Y_test, trainSize, testSize, features};
    return tts;
}

double* normalize(double* X, int samples, int features) {

    double* X_norm = (double*)malloc(sizeof(double) * samples * features);
    double* mean = (double*)malloc(sizeof(double) * features);
    double* std = (double*)malloc(sizeof(double) * features);

    // Calculate mean and std
    for (int i = 0; i < features; i++) {
        double sum = 0;
        for (int j = 0; j < samples; j++) {
            sum += X[j * features + i];
        }
        mean[i] = sum / samples;
    }

    for (int i = 0; i < features; i++) {
        double sum = 0;
        for (int j = 0; j < samples; j++) {
            sum += pow(X[j * features + i] - mean[i], 2);
        }
        std[i] = sqrt(sum / samples);
    }

    // Normalize
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            X_norm[i * features + j] = (X[i * features + j] - mean[j]) / std[j];
        }
    }

    return X_norm;
}

double* one_hot_encode(double* Y, int samples, int classes) {

    double* Y_encoded = (double*)malloc(sizeof(double) * samples * classes);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < classes; j++) {
            if (Y[i] == j) {
                Y_encoded[i * classes + j] = 1;
            } else {
                Y_encoded[i * classes + j] = 0;
            }
        }
    }

    return Y_encoded;
}

double accuracy_score(double* y_true, double* y_pred, int rows, int cols) {

    double* y_true_argmax = argmax(y_true, rows, cols);
    double* y_pred_argmax = argmax(y_pred, rows, cols);

    double correct = 0;
    for (int i = 0; i < rows; i++) {
        if (y_true_argmax[i] == y_pred_argmax[i]) {
            correct++;
        }
    }

    return correct / rows;
}


#endif
