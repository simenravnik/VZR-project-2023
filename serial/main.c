#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "helper.h"

struct MLP_model {
    double** W1;
    double** b1; 
    double** W2; 
    double** b2;
};
typedef struct MLP_model MLP_model;

double** allocate_matrix(int rows, int cols) {
    double** mat = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = malloc(cols * sizeof(double));
    }
    return mat;
}

void free_matrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

double** random_matrix(int rows, int cols) {
    double** mat = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return mat;
}

double** ones_matrix(int rows, int cols) {
    double** mat = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = 1.0;
        }
    }
    return mat;
}

double** matrix_transpose(double** mat, int rows, int cols) {
    double** mat_T = allocate_matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat_T[j][i] = mat[i][j];
        }
    }
    return mat_T;
}

double** matrix_multiply(double** mat1, double** mat2, int rows1, int cols1, int rows2, int cols2) {
    double** result = allocate_matrix(rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1[i][k] * mat2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

double** matrix_multiply_scalar(double** mat, int rows, int cols, double scalar) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
    return result;
}

double** matrix_hadamard(double** mat1, double** mat2, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    return result;
}

double** matrix_squaring(double** mat, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = pow(mat[i][j], 2);
        }
    }
    return result;
}

double** matrix_add(double** mat, double** vec, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat[i][j] + vec[0][j];
        }
    }
    return result;
}

double** matrix_subtract(double** mat1, double** mat2, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

double** matrix_tanh(double** mat, int rows, int cols) {
    double** result = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        result[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            result[i][j] = tanh(mat[i][j]);
        }
    }
    return result;
}

MLP_model train_mlp(double** X, double** Y, int samples, int features, int outputs, int hiddenSize, double alpha, int batchSize, int epochs) {
    // Initialize weights and biases
    double** W1 = random_matrix(features, hiddenSize);
    double** b1 = random_matrix(1, hiddenSize); // Vector
    double** W2 = random_matrix(hiddenSize, outputs);
    double** b2 = random_matrix(1, outputs); // Vector

    int batches = (int)ceil((double)samples / batchSize);

    printf("Training...\n");
    fflush(stdout);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < batches; b++) {
            // Generate random batch indices
            int* batch_rows = (int*)malloc(batchSize * sizeof(int));
            for (int i = 0; i < batchSize; i++) {
                batch_rows[i] = rand() % samples;
            }

            // Extract batch inputs and targets
            double** X_b = allocate_matrix(batchSize, features);
            double** Y_b = allocate_matrix(batchSize, outputs);
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    X_b[i][j] = X[batch_rows[i]][j];
                }
                for (int j = 0; j < outputs; j++) {
                    Y_b[i][j] = Y[batch_rows[i]][j];
                }
            }

            // Calculate hidden layer activations
            double** H = matrix_tanh(matrix_add(matrix_multiply(X_b, W1, batchSize, features, features, hiddenSize), b1, batchSize, hiddenSize), batchSize, hiddenSize); // batchSize x hiddenSize
            double** Y_hat = matrix_tanh(matrix_add(matrix_multiply(H, W2, batchSize, hiddenSize, hiddenSize, outputs), b2, batchSize, outputs), batchSize, outputs); // batchSize x outputs

            // Calculate error
            double** E = matrix_subtract(Y_hat, Y_b, batchSize, outputs); // batchSize x outputs

            // Calculate gradients for the second layer
            double** w2_g = matrix_multiply(matrix_transpose(H, batchSize, hiddenSize), matrix_hadamard(E, matrix_subtract(ones_matrix(batchSize, outputs), matrix_squaring(Y_hat, batchSize, outputs), batchSize, outputs), batchSize, outputs), hiddenSize, batchSize, batchSize, outputs); // hiddenSize x outputs
            double** b2_g = matrix_multiply(ones_matrix(1, batchSize), matrix_hadamard(E, matrix_subtract(ones_matrix(batchSize, outputs), matrix_squaring(Y_hat, batchSize, outputs), batchSize, outputs), batchSize, outputs), 1, batchSize, batchSize, outputs);   // 1 x outputs

            // Calculate gradients for the first layer
            double** H_e = matrix_multiply(matrix_hadamard(E, matrix_subtract(ones_matrix(batchSize, outputs), matrix_squaring(Y_hat, batchSize, outputs), batchSize, outputs), batchSize, outputs), matrix_transpose(W2, hiddenSize, outputs), batchSize, outputs, outputs, hiddenSize); // batchSize x hiddenSize
            H_e = matrix_hadamard(H_e, matrix_subtract(ones_matrix(batchSize, hiddenSize), matrix_squaring(H, batchSize, hiddenSize), batchSize, hiddenSize), batchSize, hiddenSize); // batchSize x hiddenSize
            
            double** W1_g = matrix_multiply(matrix_transpose(X_b, batchSize, features), H_e, features, batchSize, batchSize, hiddenSize); // features x hiddenSize
            double** b1_g = matrix_multiply(ones_matrix(1, batchSize), H_e, 1, batchSize, batchSize, hiddenSize); // 1 x hiddenSize

            // Update weights and biases
            W1 = matrix_subtract(W1, matrix_multiply_scalar(W1_g, features, hiddenSize, alpha), features, hiddenSize);  // features x hiddenSize
            b1 = matrix_subtract(b1, matrix_multiply_scalar(b1_g, 1, hiddenSize, alpha), 1, hiddenSize);    // 1 x hiddenSize
            W2 = matrix_subtract(W2, matrix_multiply_scalar(w2_g, hiddenSize, outputs, alpha), hiddenSize, outputs); // hiddenSize x outputs
            b2 = matrix_subtract(b2, matrix_multiply_scalar(b2_g, 1, outputs, alpha), 1, outputs);  // 1 x outputs

            // Free memory
            free(batch_rows);
            free_matrix(X_b, batchSize);
            free_matrix(Y_b, batchSize);
            free_matrix(H, batchSize);
            free_matrix(Y_hat, batchSize);
            free_matrix(E, batchSize);
            free_matrix(w2_g, hiddenSize);
            free_matrix(b2_g, 1);
            free_matrix(H_e, batchSize);
            free_matrix(W1_g, features);
            free_matrix(b1_g, 1);


            if (epoch % 100 == 0) {
                if (b == batches - 1) {
                    printf("Epoch %d/%d, Batch %d/%d\n", epoch + 1, epochs, b + 1, batches);
                    fflush(stdout);
                }
            }
        }
    }

    MLP_model model;
    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.b2 = b2;

    return model;
}

int main(int argc, char** argv) {

    DataFrame data = read_csv("../data/heart-processed.csv");
    // data = data.sample(frac=1); // shuffle the data

    // print_data_frame(data);

    double **X = (double **)malloc(data.rows * sizeof(double *));
    for (int i = 0; i < data.rows; i++) {
        X[i] = (double *)malloc((data.cols - 1) * sizeof(double));
    }

    double **Y = (double **)malloc(data.rows * sizeof(double *));
    for (int i = 0; i < data.rows; i++) {
        Y[i] = (double *)malloc(sizeof(double));
    }

    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols - 1; j++) {
            X[i][j] = data.data[i][j];
        }
        Y[i][0] = data.data[i][data.cols - 1];
    }


    printf("Shape of X: %d, %d\n", data.rows, data.cols - 1);
    printf("Shape of Y: %d, %d\n", data.rows, 1);

    // Train-test split
    int train_size = (int)(data.rows * 0.8);
    int test_size = data.rows - train_size;
    
    double **X_train = (double **)malloc(train_size * sizeof(double *));
    double **X_test = (double **)malloc(test_size * sizeof(double *));
    double **Y_train = (double **)malloc(train_size * sizeof(double *));
    double **Y_test = (double **)malloc(test_size * sizeof(double *));

    // Assign values to X_train and X_test
    for (int i = 0; i < train_size; i++) {
        X_train[i] = (double *)malloc((data.cols - 1) * sizeof(double));
        for (int j = 0; j < data.cols - 1; j++) {
            X_train[i][j] = X[i][j];
        }
    }

    for (int i = 0; i < test_size; i++) {
        X_test[i] = (double *)malloc((data.cols - 1) * sizeof(double));
        for (int j = 0; j < data.cols - 1; j++) {
            X_test[i][j] = X[train_size + i][j];
        }
    }

    // Assign values to Y_train and Y_test
    for (int i = 0; i < train_size; i++) {
        Y_train[i] = (double *)malloc(sizeof(double));
        Y_train[i][0] = Y[i][0];
    }

    for (int i = 0; i < test_size; i++) {
        Y_test[i] = (double *)malloc(sizeof(double));
        Y_test[i][0] = Y[train_size + i][0];
    }

    printf("Shape of X_train: %d, %d\n", train_size, data.cols - 1);
    printf("Shape of X_test: %d, %d\n", test_size, data.cols - 1);
    printf("Shape of Y_train: %d, %d\n", train_size, 1);
    printf("Shape of Y_test: %d, %d\n", test_size, 1);
    fflush(stdout);

    int hiddenSize = 10;
    double alpha = 0.001;
    int batchSize = 5;
    int epochs = 1000;
    int features = data.cols - 1;

    // Train the model
    MLP_model model = train_mlp(X_train, Y_train, train_size, features, 1, hiddenSize, alpha, batchSize, epochs);

    // Test the model
    double** H = matrix_tanh(matrix_add(matrix_multiply(X_test, model.W1, test_size, features, features, hiddenSize), model.b1, test_size, hiddenSize), test_size, hiddenSize);
    double** Y_hat = matrix_tanh(matrix_add(matrix_multiply(H, model.W2, test_size, hiddenSize, hiddenSize, 1), model.b2, test_size, 1), test_size, 1); // batchSize x outputs

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        printf("Y_hat: %lf, Y_test: %lf\n", Y_hat[i][0], Y_test[i][0]);
        if (Y_hat[i][0] >= 0.5 && Y_test[i][0] == 1) {
            correct++;
        } else if (Y_hat[i][0] < 0.5 && Y_test[i][0] == 0) {
            correct++;
        }
    }

    printf("Accuracy: %lf\n", (double)correct / test_size);

    // Free memory
    for (int i = 0; i < data.rows; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);

    return 0;
}