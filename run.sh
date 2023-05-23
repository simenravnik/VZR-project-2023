#!/bin/bash
module load CUDA/10.1.243-GCC-8.3.0

# Library files
lib=(
    "lib/matrix/matrix.c"
    "lib/matrix/matrix_cuda.cu"
    "lib/helpers/helpers.c"
    "lib/read/read.c"
) 

# Train files
train=(
    "src/serial/train_mlp_serial.c"
)

# Compile and run
nvcc -lm -o train train.c "${lib[@]}" "${train[@]}"
srun --reservation=fri train > results/results.txt