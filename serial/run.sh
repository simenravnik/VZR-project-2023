gcc -lm -o train train.c
srun --reservation=fri train > ../results/train.txt