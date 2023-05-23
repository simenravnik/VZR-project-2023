#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0

nvcc train.cu -O2 -lm -o train
srun --reservation=fri -G1 -n1 train > results/results.txt