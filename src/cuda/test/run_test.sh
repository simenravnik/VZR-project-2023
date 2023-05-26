#!/bin/sh
module load CUDA

nvcc cuda_matrix_test.cu -O2 -lm -o cuda_matrix_test.bin
srun --reservation=fri-vr --partition=gpu --gpus=1 cuda_matrix_test.bin
