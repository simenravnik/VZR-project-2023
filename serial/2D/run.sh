gcc -lm train.c -o train
srun --reservation=fri train > ../../results/serial_2D.txt