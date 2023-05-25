#ifndef MATRIX_NEW_H
#define MATRIX_NEW_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"

void dot_new(Matrix mat1, Matrix mat2, Matrix product);
void add_new(Matrix mat1, Matrix mat2);
void subtract_new(Matrix mat1, Matrix mat2, Matrix difference);
void hadamard_new(Matrix mat1, Matrix mat2, Matrix product);
void transpose_new(Matrix mat, Matrix trans);
void sum_new(Matrix mat, Matrix sum);
void square_new(Matrix mat);
void matrix_tanh_new(Matrix mat);
void scalar_multiply_new(Matrix mat, float scalar);

void dot_new(Matrix mat1, Matrix mat2, Matrix product) {
    if (mat1.cols != mat2.rows) {
        printf("Error: Matrix dimensions do not match for dot product\n");
    }
    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            double sum = 0;
            for (int k = 0; k < mat1.cols; k++) {
                sum += mat1.data[i * mat1.cols + k] * mat2.data[k * mat2.cols + j];
            }
            product.data[i * mat2.cols + j] = (float)sum;
        }
    }
}

/**
 * Addition between matrix and vector!!!
 * Adds mat2 to each column of mat1
 * 
 * Do not use for matrix addition
*/
void add_new(Matrix mat1, Matrix mat2) {
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        mat1.data[i] = mat1.data[i] + mat2.data[i % mat1.cols];
    }
}

void subtract_new(Matrix mat1, Matrix mat2, Matrix difference) {
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        difference.data[i] = mat1.data[i] - mat2.data[i];
    }
}

void hadamard_new(Matrix mat1, Matrix mat2, Matrix product) {
    for (int i = 0; i < mat1.rows * mat1.cols; i++) {
        product.data[i] = mat1.data[i] * mat2.data[i];
    }
}

void transpose_new(Matrix mat, Matrix trans) {
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        trans.data[(i % mat.cols) * mat.rows + (i / mat.cols)] = mat.data[i];
    }
}

void sum_new(Matrix mat, Matrix sum) {
    for (int i = 0; i < mat.cols; i++) {
        double colSum = 0;
        for (int j = 0; j < mat.rows; j++) {
            colSum += mat.data[j * mat.cols + i];
        }
        sum.data[i] = (float)colSum;
    }
}

void square_new(Matrix mat) {
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        mat.data[i] = mat.data[i] * mat.data[i];
    }
}

void matrix_tanh_new(Matrix mat) {
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        mat.data[i] = tanhf(mat.data[i]);
    }
}

void scalar_multiply_new(Matrix mat, float scalar) {
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        mat.data[i] = mat.data[i] * scalar;
    }
}

#endif
