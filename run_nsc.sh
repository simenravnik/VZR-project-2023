#!/bin/sh

# SERIAL
gcc -O2 -lm -fopenmp -o train_serial.bin src/serial/train.c
srun --reservation=fri train_serial.bin > results/serial.txt
rm train_serial.bin

# OPENMP
gcc -O2 -lm -fopenmp -o train_openmp.bin src/openmp/train.c
srun --reservation=fri -n1 --cpus-per-task=8 train_openmp.bin > results/openmp.txt
rm train_openmp.bin

# CUDA
module load CUDA/10.1.243-GCC-8.3.0
nvcc -O2 -lm -o train_cuda.bin src/cuda/train.cu
srun --reservation=fri -G1 -n1 train_cuda.bin cuda > results/cuda.txt
srun --reservation=fri -G1 -n1 train_cuda.bin new > results/cuda_sk.txt
rm train_cuda.bin

# MPI
# module load OpenMPI/4.1.0-GCC-10.2.0
# mpicc src/mpi/train.c -O2 -lm -fopenmp -o train_mpi.bin
# srun --mpi=pmix -n4 -N1 --reservation=fri train_mpi.bin > results/mpi.txt
# rm train_mpi.bin