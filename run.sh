#!/bin/bash

lib=(
    "lib/matrix/matrix.c"
    "lib/helpers/helpers.c"
    "lib/read/read.c"
) 

train=(
    "src/serial/train_mlp_serial.c"
)

gcc -lm -o train train.c "${lib[@]}" "${train[@]}"
srun --reservation=fri train > results/results.txt