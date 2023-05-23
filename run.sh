#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0

train=(
    "src/serial/train_mlp_serial.c"
)

nvcc -O2 -lm -o train train.cu -include "${train[@]}"
srun --reservation=fri -G1 -n1 train > results/results.txt