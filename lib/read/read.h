#ifndef READ_H
#define READ_H

typedef struct DataFrame {
    float *data;
    int rows;
    int cols;
} DataFrame;

DataFrame read_csv(const char *filename);
void print_data_frame(DataFrame df);
void free_data_frame(DataFrame df);

#endif
