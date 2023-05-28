#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0

# SERIAL
gcc -O2 -lm -fopenmp -o train_serial.bin src/serial/train.c
srun --reservation=fri train_serial.bin > results/Serial.txt

# OPENMP
gcc -O2 -lm -fopenmp -o train_openmp.bin src/openmp/train.c
srun --reservation=fri -n1 --cpus-per-task=8 train_openmp.bin > results/OpenMP.txt

# CUDA
nvcc -O2 -lm -o train_cuda.bin src/cuda/train.cu
srun --reservation=fri -G1 -n1 train_cuda.bin cuda > results/CUDA.txt
srun --reservation=fri -G1 -n1 train_cuda.bin new > results/CUDA_SINGLE_KERNEL.txt

rm train_serial.bin train_openmp.bin train_cuda.bin train_mpi.bin

# MPI
module load OpenMPI/4.1.0-GCC-10.2.0
mpicc src/mpi/train.c -O2 -lm -fopenmp -o train_mpi.bin
srun --mpi=pmix -n4 -N1 --reservation=fri train_mpi.bin > results/MPI.txt
rm train_mpi.bin