#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct Matrix {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix allocate_matrix(int rows, int cols);
Matrix duplicate_matrix(Matrix mat);
void free_matrix(Matrix mat);
Matrix random_matrix(int rows, int cols);
void print_matrix(Matrix mat);
Matrix slice_matrix(Matrix mat, int startRow, int endRow, int startCol, int endCol);
Matrix ones(int rows, int cols);
Matrix argmax(Matrix mat);
int compare_matrices(Matrix mat1, Matrix mat2);

Matrix allocate_matrix(int rows, int cols) {
    float* data = (float*) malloc(rows * cols * sizeof(float));
    Matrix mat = {data, rows, cols};
    return mat;
}

Matrix duplicate_matrix(Matrix mat) {
    Matrix dup = allocate_matrix(mat.rows, mat.cols);
    memcpy(dup.data, mat.data, mat.rows * mat.cols * sizeof(float));
    return dup;
}

void free_matrix(Matrix mat) {
    free(mat.data);
}

Matrix random_matrix(int rows, int cols) {
    srand(1);  // Replace this with srand(time(0)); if you want to randomize the seed
    Matrix mat = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        mat.data[i] = (float) rand() / RAND_MAX;
    }
    return mat;
}

void print_matrix(Matrix mat) {
    printf("[");
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        if (i % mat.cols == 0) {
            printf("[");
        }
        printf("%f", mat.data[i]);
        if (i % mat.cols == mat.cols - 1) {
            printf("]");
        } else {
            printf(", ");
        }
        if ((i + 1) % mat.cols == 0 && i != mat.rows * mat.cols - 1) {
            printf(",\n");
        }
    }
    printf("]\n\n");
}

void print_size(Matrix mat) {
    printf("(%d, %d)\n", mat.rows, mat.cols);
}

Matrix slice_matrix(Matrix mat, int startRow, int endRow, int startCol, int endCol) {
    int rows = endRow - startRow;
    int cols = endCol - startCol;
    Matrix slice = allocate_matrix(rows, cols);
    for (int i = startRow; i < endRow; i++) {
        for (int j = startCol; j < endCol; j++) {
            slice.data[(i - startRow) * cols + (j - startCol)] = mat.data[i * endCol + j];
        }
    }
    return slice;
}

Matrix ones(int rows, int cols) {
    Matrix ones = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        ones.data[i] = 1;
    }
    return ones;
}

Matrix argmax(Matrix mat) {
    Matrix max = allocate_matrix(mat.rows, 1);
    for (int i = 0; i < mat.rows; i++) {
        max.data[i] = 0;
        for (int j = 0; j < mat.cols; j++) {
            if (mat.data[i * mat.cols + j] > mat.data[i * mat.cols + (int)max.data[i]]) {
                max.data[i] = j;
            }
        }
    }
    return max;
}

int compare_matrices(Matrix A, Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        return 1;
    }

    for (int i = 0; i < A.rows * A.cols; i++) {
        if (abs(A.data[i] - B.data[i]) > 0.0001) {
            return 1;
        }
    }

    return 0;
}

#endif
