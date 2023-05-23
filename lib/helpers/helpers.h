#ifndef HELPERS_H
#define HELPERS_H

#include "../read/read.h"
#include "../matrix/matrix.h"

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

TrainTestSplit train_test_split(DataFrame df, int trainSize, int testSize);
Data extract_data(DataFrame df);
TrainTestSplit split(float* X, float* Y, int samples, int features, int trainSize, int testSize);
Matrix normalize(Matrix X);
Matrix one_hot_encode(Matrix Y, int classes);
float accuracy_score(Matrix y_true, Matrix y_pred);

#endif
