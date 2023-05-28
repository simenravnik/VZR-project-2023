/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "../../parameters.h"
#include "../../lib/read/read.h"
#include "../../lib/matrix/matrix.h"
#include "../../lib/helpers/helpers.h"
#include "../../lib/models/mlp_model.h"

#include "train_openmp.h"

int main(int argc, char** argv) {

    // Prepare dataset
    TrainTestSplit split = prepare_dataset(FILEPATH, CLASSES, TRAIN_SIZE_PERCENTAGE);

    double elapsed = omp_get_wtime();
    MLP_model model = train_openmp(split.X_train, split.Y_train, HIDDEN_SIZE, ETA, BATCH_SIZE, EPOCHS);
    elapsed = omp_get_wtime() - elapsed;
    printf("Time: %0.3f milliseconds \n", elapsed * 1000);

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