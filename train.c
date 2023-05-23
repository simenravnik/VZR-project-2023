/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lib/read/read.h"
#include "lib/helpers/helpers.h"

#include "src/serial/train_mlp_serial.h"

int main(int argc, char** argv) {

    // Read data
    DataFrame df = read_csv("data/iris.data");

    // Train-test split
    int trainSize = (int)(df.rows * 1.0);   // TODO: For now we use the entire dataset for trainingc
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
    MLP_model model = train_mlp_serial(split.X_train, split.Y_train, hiddenSize, eta, batchSize, epochs);

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
