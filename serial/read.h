#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct DataFrame {
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