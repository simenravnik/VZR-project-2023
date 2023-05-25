#!/bin/sh
module load CUDA

train=(
    "src/serial/train_mlp_serial.c"
)

nvcc -O2 -lm -o train.bin train.cu -include "${train[@]}"

srun --reservation=fri-vr --partition=gpu --gpus=1 train.bin serial > results/Serial.txt
srun --reservation=fri-vr --partition=gpu --gpus=1 train.bin cuda > results/CUDA.txt
srun --reservation=fri-vr --partition=gpu --gpus=1 train.bin cuda_new > results/CUDA_NEW.txt