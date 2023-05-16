#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double** allocate_matrix(int rows, int cols) {
    double** mat = (double**) malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*) malloc(cols * sizeof(double));
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

void print_matrix(double** mat, int rows, int cols) {
    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%f", mat[i][j]);
            if (j < cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < rows - 1) {
            printf(",\n");
        }
    }
    printf("]\n\n");
}

double** slice_matrix(double** mat, int startRow, int endRow, int startCol, int endCol) {
    int rows = endRow - startRow;
    int cols = endCol - startCol;

    double** slice = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            slice[i][j] = mat[startRow + i][startCol + j];
        }
    }
    return slice;
}

double** dot(double** mat1, double** mat2, int rows1, int cols1, int rows2, int cols2) {
    double** result = allocate_matrix(rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

double** add(double** mat1, double** mat2, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = mat1[i][j] + mat2[0][j];
        }
    }
    return result;
}

double** matrix_tanh(double** mat, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = tanh(mat[i][j]);
        }
    }
    return result;
}

double** subtract(double** mat1, double** mat2, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = mat1[i][j] - mat2[0][j];
        }
    }
    return result;
}

double** transpose(double** mat, int rows, int cols) {
    double** result = allocate_matrix(cols, rows);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++){
            result[i][j] = mat[j][i];
        }
    }
    return result;
}

double** hadamard(double** mat1, double** mat2, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    return result;
}

double** square(double** mat, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = mat[i][j] * mat[i][j];
        }
    }
    return result;
}

double** ones(int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = 1;
        }
    }
    return result;
}

double** sum(double** mat, int rows, int cols) {
    double** result = allocate_matrix(1, cols);
    for (int i = 0; i < cols; i++) {
        result[0][i] = 0;
        for (int j = 0; j < rows; j++){
            result[0][i] += mat[j][i];
        }
    }
    return result;
}

double** scalar_multiply(double** mat, double scalar, int rows, int cols) {
    double** result = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            result[i][j] = mat[i][j] * scalar;
        }
    }
    return result;
}

double** argmax(double** mat, int rows, int cols) {
    double** result = allocate_matrix(rows, 1);
    for (int i = 0; i < rows; i++) {
        result[i][0] = 0;
        for (int j = 0; j < cols; j++){
            if (mat[i][j] > mat[i][(int)result[i][0]]) {
                result[i][0] = j;
            }
        }
    }

    return result;
}

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