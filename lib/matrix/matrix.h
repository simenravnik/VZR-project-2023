#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct Matrix {
    float* data;
    int rows;
    int cols;
} Matrix;

Matrix allocate_matrix(int rows, int cols);
void free_matrix(Matrix mat);
Matrix random_matrix(int rows, int cols);
void print_matrix(Matrix mat);
Matrix slice_matrix(Matrix mat, int startRow, int endRow, int startCol, int endCol);
Matrix dot(Matrix mat1, Matrix mat2);
Matrix add(Matrix mat1, Matrix mat2);
Matrix subtract(Matrix mat1, Matrix mat2);
Matrix hadamard(Matrix mat1, Matrix mat2);
Matrix transpose(Matrix mat);
Matrix sum(Matrix mat);
Matrix ones(int rows, int cols);
Matrix square(Matrix mat);
Matrix matrix_tanh(Matrix mat);
Matrix scalar_multiply(Matrix mat, float scalar);
Matrix argmax(Matrix mat);
int compare_matrices(Matrix mat1, Matrix mat2);

Matrix allocate_matrix(int rows, int cols) {
    float* data = (float*) malloc(rows * cols * sizeof(float));
    Matrix mat = {data, rows, cols};
    return mat;
}

void free_matrix(Matrix mat) {
    free(mat.data);
}

Matrix random_matrix(int rows, int cols) {
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

Matrix dot(Matrix mat1, Matrix mat2) {
    if (mat1.cols != mat2.rows) {
        printf("Error: Matrix dimensions do not match for dot product\n");
        Matrix null = {NULL, 0, 0};
        return null;
    }
    Matrix product = allocate_matrix(mat1.rows, mat2.cols);
    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            double sum = 0;
            for (int k = 0; k < mat1.cols; k++) {
                sum += mat1.data[i * mat1.cols + k] * mat2.data[k * mat2.cols + j];
            }
            product.data[i * mat2.cols + j] = (float)sum;
        }
    }
    return product;
}

/**
 * Addition between matrix and vector!!!
 * Adds mat2 to each column of mat1
 * 
 * Do not use for matrix addition
*/
Matrix add(Matrix mat1, Matrix mat2) {
    Matrix sum = allocate_matrix(mat1.rows, mat1.cols);
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        sum.data[i] = mat1.data[i] + mat2.data[i % mat1.cols];
    }
    return sum;
}

Matrix subtract(Matrix mat1, Matrix mat2) {
    Matrix difference = allocate_matrix(mat1.rows, mat1.cols);
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        difference.data[i] = mat1.data[i] - mat2.data[i];
    }
    return difference;
}

Matrix hadamard(Matrix mat1, Matrix mat2) {
    Matrix product = allocate_matrix(mat1.rows, mat1.cols);
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        product.data[i] = mat1.data[i] * mat2.data[i];
    }
    return product;
}

Matrix transpose(Matrix mat) {
    Matrix trans = allocate_matrix(mat.cols, mat.rows);
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        trans.data[(i % mat.cols) * mat.rows + (i / mat.cols)] = mat.data[i];
    }
    return trans;
}

Matrix sum(Matrix mat) {
    Matrix sum = allocate_matrix(1, mat.cols);
    for (int i = 0; i < mat.cols; i++) {
        double colSum = 0;
        for (int j = 0; j < mat.rows; j++) {
            colSum += mat.data[j * mat.cols + i];
        }
        sum.data[i] = (float)colSum;
    }
    return sum;
}

Matrix ones(int rows, int cols) {
    Matrix ones = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        ones.data[i] = 1;
    }
    return ones;
}

Matrix square(Matrix mat) {
    Matrix square = allocate_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        square.data[i] = mat.data[i] * mat.data[i];
    }
    return square;
}

Matrix matrix_tanh(Matrix mat) {
    Matrix tanh = allocate_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        tanh.data[i] = tanhf(mat.data[i]);
    }
    return tanh;
}

Matrix scalar_multiply(Matrix mat, float scalar) {
    Matrix product = allocate_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        product.data[i] = mat.data[i] * scalar;
    }
    return product;
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
