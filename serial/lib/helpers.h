#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "read.h"
#include "matrix.h"

typedef struct TrainTestSplit {
    double** X_train;
    double** X_test;
    double** Y_train;
    double** Y_test;
    int trainSize;
    int testSize;
    int features;
} TrainTestSplit;

typedef struct Data {
    double** X;
    double** Y;
} Data;

TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize);
Data extract_data(DataFrame df);
TrainTestSplit split(double** X, double** Y, int samples, int features, int trainSize, int testSize);
double** normalize(double** X, int samples, int features);
double** one_hot_encode(double** Y, int samples, int classes);
double accuracy_score(double** y_true, double** y_pred, int rows, int cols);

/**
 * Returns data splitted into train and test
*/
TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize) {

    // Split data into features and labels
    Data data = extract_data(df);

    // Split data into train and test
    TrainTestSplit tts = split(data.X, data.Y, df.rows, df.cols - 1, trainSize, testSize);

    return tts;
}

/**
 * Extracts data from a DataFrame and returns a Data struct
*/
Data extract_data(DataFrame df) {
    Data data;
    data.X = (double**)malloc(df.rows * sizeof(double*));
    for (int i = 0; i < df.rows; i++) {
        data.X[i] = (double*)malloc((df.cols - 1) * sizeof(double));
    }

    data.Y = (double**)malloc(df.rows * sizeof(double*));
    for (int i = 0; i < df.rows; i++) {
        data.Y[i] = (double*)malloc(sizeof(double));
    }

    for (int i = 0; i < df.rows; i++) {
        for (int j = 0; j < df.cols - 1; j++) {
            data.X[i][j] = df.data[i][j];
        }
        data.Y[i][0] = df.data[i][df.cols - 1];
    }
    return data;
}

/**
 * Splits data into train and test
*/
TrainTestSplit split(double** X, double** Y, int samples, int features, int trainSize, int testSize) {
    double** X_train = (double**)malloc(trainSize * sizeof(double*));
    double** X_test = (double**)malloc(testSize * sizeof(double*));
    double** Y_train = (double**)malloc(trainSize * sizeof(double*));
    double** Y_test = (double**)malloc(testSize * sizeof(double*));

    // Assign values to X_train and X_test
    for (int i = 0; i < trainSize; i++) {
        X_train[i] = (double*)malloc(features * sizeof(double));
        for (int j = 0; j < features; j++) {
            X_train[i][j] = X[i][j];
        }
    }

    for (int i = 0; i < testSize; i++) {
        X_test[i] = (double*)malloc(features * sizeof(double));
        for (int j = 0; j < features; j++) {
            X_test[i][j] = X[trainSize + i][j];
        }
    }

    // Assign values to Y_train and Y_test
    for (int i = 0; i < trainSize; i++) {
        Y_train[i] = (double*)malloc(sizeof(double));
        Y_train[i][0] = Y[i][0];
    }

    for (int i = 0; i < testSize; i++) {
        Y_test[i] = (double*)malloc(sizeof(double));
        Y_test[i][0] = Y[trainSize + i][0];
    }

    TrainTestSplit tts = {X_train, X_test, Y_train, Y_test, trainSize, testSize, features};
    return tts;
}

double** normalize(double** X, int samples, int features) {
    double** X_normalized = (double**)malloc(samples * sizeof(double*));
    for (int i = 0; i < samples; i++) {
        X_normalized[i] = (double*)malloc(features * sizeof(double));
    }

    double* mean = (double*)malloc(features * sizeof(double));
    double* std = (double*)malloc(features * sizeof(double));

    // Calculate mean and std
    for (int i = 0; i < features; i++) {
        double sum = 0;
        for (int j = 0; j < samples; j++) {
            sum += X[j][i];
        }
        mean[i] = sum / samples;
    }

    for (int i = 0; i < features; i++) {
        double sum = 0;
        for (int j = 0; j < samples; j++) {
            sum += pow(X[j][i] - mean[i], 2);
        }
        std[i] = sqrt(sum / samples);
    }

    // Normalize
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            X_normalized[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }

    return X_normalized;
}

double** one_hot_encode(double** Y, int samples, int classes) {
    double** Y_encoded = (double**)malloc(samples * sizeof(double*));
    for (int i = 0; i < samples; i++) {
        Y_encoded[i] = (double*)malloc(classes * sizeof(double));
        for (int j = 0; j < classes; j++) {
            Y_encoded[i][j] = 0;
        }
        Y_encoded[i][(int)Y[i][0]] = 1;
    }

    return Y_encoded;
}

// Additional helper functions
double accuracy_score(double** y_true, double** y_pred, int rows, int cols) {
    double** y_true_argmax = argmax(y_true, rows, cols);
    double** y_pred_argmax = argmax(y_pred, rows, cols);

    double correct = 0;
    for (int i = 0; i < rows; i++) {
        if (y_true_argmax[i][0] == y_pred_argmax[i][0]) {
            correct++;
        }
    }

    return correct / rows;
}


#endif
