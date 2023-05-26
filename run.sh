#!/bin/sh
module load CUDA

# !!! THIS IS FOR ARNES !!!
# Fix if you want to run on NSC

# SERIAL
gcc -O2 -lm -fopenmp -o train_serial.bin src/serial/train.c
srun --reservation=fri-vr --partition=gpu train_serial.bin > results/Serial.txt

# OPENMP
gcc -O2 -lm -fopenmp -o train_openmp.bin src/openmp/train.c
srun --reservation=fri-vr --partition=gpu --cpus-per-task=4 train_openmp.bin > results/OpenMP.txt

# CUDA
nvcc -O2 -lm -o train_cuda.bin src/cuda/train.cu
srun --reservation=fri-vr --partition=gpu --gpus=1 train_cuda.bin cuda > results/CUDA.txt
srun --reservation=fri-vr --partition=gpu --gpus=1 train_cuda.bin new > results/CUDA_SINGLE_KERNEL.txt

# MPI
# We have to compile MPI using srun (ARNES specific) + run it using srun manually
# module load OpenMPI/4.1.1-GCC-11.2.0
# srun --reservation=fri-vr --partition=gpu mpicc -O2 -lm -fopenmp -o train_mpi.bin src/mpi/train.c
# srun --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=2 --ntasks=4 train_mpi.bin > results/MPI.txt

rm train_serial.bin train_openmp.bin train_cuda.bin