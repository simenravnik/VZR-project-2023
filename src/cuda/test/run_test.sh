#!/bin/sh

# NSC
module load CUDA/10.1.243-GCC-8.3.0
nvcc matrix_cuda_test.cu -O2 -lm -o matrix_cuda_test.bin
srun --reservation=fri -G1 -n1 matrix_cuda_test.bin
rm matrix_cuda_test.bin

# Arnes
# module load CUDA
# nvcc matrix_cuda_test.cu -O2 -lm -o matrix_cuda_test.bin
# srun --reservation=fri-vr --partition=gpu --gpus=1 matrix_cuda_test.bin
# rm matrix_cuda_test.bin
