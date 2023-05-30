#!/bin/sh

# NSC
# module load OpenMPI/4.1.0-GCC-10.2.0
# mpicc train_mpi_test.c -O2 -lm -fopenmp -o train_mpi_test.bin
# srun --mpi=pmix -n4 -N1 --reservation=fri train_mpi_test.bin
# rm train_mpi_test.bin

# ARNES
module load OpenMPI/4.1.1-GCC-11.2.0
srun --reservation=fri-vr --partition=gpu mpicc -O2 -lm -fopenmp -o train_mpi_test.bin train_mpi_test.c
srun --reservation=fri-vr --partition=gpu --mpi=pmix --nodes=1 --ntasks=4 train_mpi_test.bin
rm train_mpi_test.bin