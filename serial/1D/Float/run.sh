gcc -lm train.c -o train
srun --reservation=fri train > ../../../results/serial_1D.txt