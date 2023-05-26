/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../../parameters.h"
#include "../../lib/read/read.h"
#include "../../lib/matrix/matrix.h"
#include "../../lib/helpers/helpers.h"
#include "../../lib/matrix/cuda_matrix.h"
#include "../../lib/models/mlp_model.h"
#include "../../lib/time/cuda_timer.h"

#include "train_mlp_cuda.h"
#include "train_mlp_cuda_new.h"

int main(int argc, char** argv) {

    // Prepare dataset
    TrainTestSplit split = prepare_dataset(FILEPATH, CLASSES, TRAIN_SIZE_PERCENTAGE);

    MLP_model model;
    
    // Start timer
    cudaEvent_t start, stop;
    cuda_start_timer(&start, &stop);

    // Train the model
    if (strcmp(argv[1], "cuda") == 0) {
        printf("CUDA\n");
        model = train_mlp_cuda(split.X_train, split.Y_train, HIDDEN_SIZE, ETA, BATCH_SIZE, EPOCHS);
    } else if (strcmp(argv[1], "new") == 0) {
        printf("CUDA SINGLE KERNEL\n");
        model = train_mlp_cuda_new(split.X_train, split.Y_train, HIDDEN_SIZE, ETA, BATCH_SIZE, EPOCHS);
    } else {
        printf("Invalid argument. Please use either 'serial' or 'cuda'.\n");
        return 1;
    }
    
    // Stop timer
    printf("Time: %0.3f milliseconds \n", cuda_stop_timer(&start, &stop));

    // Test the model
    float accuracy;
    if (TRAIN_SIZE_PERCENTAGE == 1.0) {
        // Test on training set
        printf("Training set used for testing\n");
        accuracy = predict(split.X_train, split.Y_train, model);
    } else {
        accuracy = predict(split.X_test, split.Y_test, model);
    }
    printf("Accuracy: %f\n\n", accuracy);

    // Free memory
    free_split(split);
    free_model(model);
    
    return 0;
}