#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0

nvcc cuda_matrix_test.cu -O2 -lm -o cuda_matrix_test
srun --reservation=fri -G1 -n1 cuda_matrix_test
