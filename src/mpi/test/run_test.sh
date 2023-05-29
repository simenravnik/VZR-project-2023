#!/bin/sh

module load OpenMPI/4.1.0-GCC-10.2.0
mpicc train_mpi_test.c -O2 -lm -fopenmp -o train_mpi_test.bin
srun --mpi=pmix -n4 -N1 --reservation=fri train_mpi_test.bin
rm train_mpi_test.bin