#ifndef MATRIX_H
#define MATRIX_H

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

#endif
