gcc -lm train.c -o train
srun --reservation=fri train > ../results/train.txt