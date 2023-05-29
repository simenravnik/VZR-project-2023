#!/bin/sh

# SERIAL
gcc -O2 -lm -fopenmp -o train_serial.bin src/serial/train.c
srun --reservation=fri-vr --partition=gpu train_serial.bin > results/serial.txt
rm train_serial.bin

# OPENMP
gcc -O2 -lm -fopenmp -o train_openmp.bin src/openmp/train.c
srun --reservation=fri-vr --partition=gpu --cpus-per-task=8 train_openmp.bin > results/openmp.txt
rm train_openmp.bin

# CUDA
module load CUDA
nvcc -O2 -lm -o train_cuda.bin src/cuda/train.cu
srun --reservation=fri-vr --partition=gpu --gpus=1 train_cuda.bin cuda > results/cuda.txt
srun --reservation=fri-vr --partition=gpu --gpus=1 train_cuda.bin new > results/cuda_sk.txt
rm train_cuda.bin

# MPI
# We have to compile MPI using srun (ARNES specific) + run it using srun manually
# module load OpenMPI/4.1.1-GCC-11.2.0
# srun --reservation=fri-vr --partition=gpu mpicc -O2 -lm -fopenmp -o train_mpi.bin src/mpi/train.c
# srun --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=2 --ntasks=4 train_mpi.bin > results/mpi.txt
# rm train_mpi.bin