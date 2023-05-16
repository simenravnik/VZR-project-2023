#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct {
    double **data;
    int rows;
    int cols;
} DataFrame;

DataFrame read_csv(const char *filename) {
    DataFrame df;
    df.data = NULL;
    df.rows = 0;
    df.cols = 0;

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return df;
    }

    // Count the number of rows and columns in the CSV file
    char line[512];
    // TODO: Skip the header line
    // Count the number of rows and columns
    while (fgets(line, sizeof(line), file) != NULL) {
        df.rows++;
        if (df.cols == 0) {
            char *token = strtok(line, ",");
            while (token != NULL) {
                df.cols++;
                token = strtok(NULL, ",");
            }
        }
    }

    // Allocate memory for the data array
    df.data = (double **)malloc(df.rows * sizeof(double *));
    for (int i = 0; i < df.rows; i++) {
        df.data[i] = (double *)malloc(df.cols * sizeof(double));
    }

    // Rewind the file pointer to read data
    rewind(file);

    // Read the CSV data into the DataFrame
    int row = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        int col = 0;
        char *token = strtok(line, ",");
        while (token != NULL) {
            df.data[row][col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }

    fclose(file);
    return df;
}

void print_data_frame(DataFrame df) {
    for (int i = 0; i < df.rows; i++) {
        for (int j = 0; j < df.cols; j++) {
            printf("%f ", df.data[i][j]);
        }
        printf("\n");
    }
}

void free_data_frame(DataFrame df) {
    for (int i = 0; i < df.rows; i++) {
        free(df.data[i]);
    }
    free(df.data);
}

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
