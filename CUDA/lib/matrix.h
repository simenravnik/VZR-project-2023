#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

float* allocate_matrix(int rows, int cols);
void free_matrix(float* mat, int rows);
float* random_matrix(int rows, int cols);
void print_matrix(float* mat, int rows, int cols);
float* slice_matrix(float* mat, int startRow, int endRow, int startCol, int endCol);
float* dot(float* mat1, float* mat2, int rows1, int cols1, int rows2, int cols2);
float* add(float* mat1, float* mat2, int rows, int cols);
float* subtract(float* mat1, float* mat2, int rows, int cols);
float* hadamard(float* mat1, float* mat2, int rows, int cols);
float* transpose(float* mat, int rows, int cols);
float* sum(float* mat, int rows, int cols);
float* ones(int rows, int cols);
float* square(float* mat, int rows, int cols);
float* matrix_tanh(float* mat, int rows, int cols);
float* scalar_multiply(float* mat, float scalar, int rows, int cols);
float* argmax(float* mat, int rows, int cols);

float* allocate_matrix(int rows, int cols) {
    float* mat = (float*) malloc(rows * cols * sizeof(float));
    return mat;
}

void free_matrix(float* mat, int rows) {
    free(mat);
}

float* random_matrix(int rows, int cols) {
    float* mat = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float) rand() / RAND_MAX;
    }
    return mat;
}

void print_matrix(float* mat, int rows, int cols) {
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

float* slice_matrix(float* mat, int startRow, int endRow, int startCol, int endCol) {
    int rows = endRow - startRow;
    int cols = endCol - startCol;
    float* slice = allocate_matrix(rows, cols);
    for (int i = startRow; i < endRow; i++) {
        for (int j = startCol; j < endCol; j++) {
            slice[(i - startRow) * cols + (j - startCol)] = mat[i * endCol + j];
        }
    }
    return slice;
}

float* dot(float* mat1, float* mat2, int rows1, int cols1, int rows2, int cols2) {
    if (cols1 != rows2) {
        printf("Error: Matrix dimensions do not match for dot product\n");
        return NULL;
    }
    float* product = allocate_matrix(rows1, cols2);
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            float sum = 0;
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
float* add(float* mat1, float* mat2, int rows, int cols) {
    float* sum = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        sum[i] = mat1[i] + mat2[i % cols];
    }
    return sum;
}

float* subtract(float* mat1, float* mat2, int rows, int cols) {
    float* difference = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        difference[i] = mat1[i] - mat2[i];
    }
    return difference;
}

float* hadamard(float* mat1, float* mat2, int rows, int cols) {
    float* product = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        product[i] = mat1[i] * mat2[i];
    }
    return product;
}

float* transpose(float* mat, int rows, int cols) {
    float* trans = allocate_matrix(cols, rows);
    for (int i = 0; i < rows * cols; i++) {
        trans[(i % cols) * rows + (i / cols)] = mat[i];
    }
    return trans;
}

float* sum(float* mat, int rows, int cols) {
    float* sum = allocate_matrix(1, cols);
    for (int i = 0; i < cols; i++) {
        float colSum = 0;
        for (int j = 0; j < rows; j++) {
            colSum += mat[j * cols + i];
        }
        sum[i] = colSum;
    }
    return sum;
}

float* ones(int rows, int cols) {
    float* ones = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        ones[i] = 1;
    }
    return ones;
}

float* square(float* mat, int rows, int cols) {
    float* square = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        square[i] = mat[i] * mat[i];
    }
    return square;
}

float* matrix_tanh(float* mat, int rows, int cols) {
    float* tanh = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        tanh[i] = tanhf(mat[i]);
    }
    return tanh;
}

float* scalar_multiply(float* mat, float scalar, int rows, int cols) {
    float* product = allocate_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        product[i] = mat[i] * scalar;
    }
    return product;
}

float* argmax(float* mat, int rows, int cols) {
    float* max = allocate_matrix(rows, 1);
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
