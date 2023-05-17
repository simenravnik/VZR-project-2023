#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double* allocate_matrix(int rows, int cols);
void free_matrix(double* mat, int rows);
double* random_matrix(int rows, int cols);
void print_matrix(double* mat, int rows, int cols);
double* slice_matrix(double* mat, int startRow, int endRow, int startCol, int endCol);
double* dot(double* mat1, double* mat2, int rows1, int cols1, int rows2, int cols2);
double* add(double* mat1, double* mat2, int rows, int cols);
double* subtract(double* mat1, double* mat2, int rows, int cols);
double* hadamard(double* mat1, double* mat2, int rows, int cols);
double* transpose(double* mat, int rows, int cols);
double* sum(double* mat, int rows, int cols);
double* ones(int rows, int cols);
double* square(double* mat, int rows, int cols);
double* matrix_tanh(double* mat, int rows, int cols);
double* scalar_multiply(double* mat, double scalar, int rows, int cols);
double* argmax(double* mat, int rows, int cols);

double* allocate_matrix(int rows, int cols) {
    double* mat = (double*) malloc(rows * cols * sizeof(double));
    return mat;
}

void free_matrix(double* mat, int rows) {
    free(mat);
}

double* random_matrix(int rows, int cols) {
    double* mat = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (double) rand() / RAND_MAX;
    }
    return mat;
}

void print_matrix(double* mat, int rows, int cols) {
    printf("[");
    for (int i = 0; i < rows * cols; i++) {
        if (i % cols == 0) {
            printf("[");
        }
        printf("%f", mat[i]);
        if (i % cols == cols - 1) {
            printf("]");
        } else {
            printf(", ");
        }
        if ((i + 1) % cols == 0 && i != rows * cols - 1) {
            printf(",\n");
        }
    }
    printf("]\n\n");
}

double* slice_matrix(double* mat, int startRow, int endRow, int startCol, int endCol) {
    int rows = endRow - startRow;
    int cols = endCol - startCol;
    double* slice = allocate_matrix(rows, cols);
    for (int i = startRow; i < endRow; i++) {
        for (int j = startCol; j < endCol; j++) {
            slice[(i - startRow) * cols + (j - startCol)] = mat[i * endCol + j];
        }
    }
    return slice;
}

double* dot(double* mat1, double* mat2, int rows1, int cols1, int rows2, int cols2) {
    if (cols1 != rows2) {
        printf("Error: Matrix dimensions do not match for dot product\n");
        return NULL;
    }
    double* product = allocate_matrix(rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            double sum = 0;
            for (int k = 0; k < cols1; k++) {
                sum += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
            product[i * cols2 + j] = sum;
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
double* add(double* mat1, double* mat2, int rows, int cols) {
    double* sum = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        sum[i] = mat1[i] + mat2[i % cols];
    }
    return sum;
}

double* subtract(double* mat1, double* mat2, int rows, int cols) {
    double* difference = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        difference[i] = mat1[i] - mat2[i];
    }
    return difference;
}

double* hadamard(double* mat1, double* mat2, int rows, int cols) {
    double* product = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        product[i] = mat1[i] * mat2[i];
    }
    return product;
}

double* transpose(double* mat, int rows, int cols) {
    double* trans = allocate_matrix(cols, rows);
    for (int i = 0; i < rows * cols; i++) {
        trans[(i % cols) * rows + (i / cols)] = mat[i];
    }
    return trans;
}

double* sum(double* mat, int rows, int cols) {
    double* sum = allocate_matrix(1, cols);
    for (int i = 0; i < cols; i++) {
        double colSum = 0;
        for (int j = 0; j < rows; j++) {
            colSum += mat[j * cols + i];
        }
        sum[i] = colSum;
    }
    return sum;
}

double* ones(int rows, int cols) {
    double* ones = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        ones[i] = 1;
    }
    return ones;
}

double* square(double* mat, int rows, int cols) {
    double* square = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        square[i] = mat[i] * mat[i];
    }
    return square;
}

double* matrix_tanh(double* mat, int rows, int cols) {
    double* tanh = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        tanh[i] = tanhf(mat[i]);
    }
    return tanh;
}

double* scalar_multiply(double* mat, double scalar, int rows, int cols) {
    double* product = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        product[i] = mat[i] * scalar;
    }
    return product;
}

double* argmax(double* mat, int rows, int cols) {
    double* max = allocate_matrix(rows, 1);
    for (int i = 0; i < rows; i++) {
        max[i] = 0;
        for (int j = 0; j < cols; j++) {
            if (mat[i * cols + j] > mat[i * cols + (int)max[i]]) {
                max[i] = j;
            }
        }
    }
    return max;
}

#endif
