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