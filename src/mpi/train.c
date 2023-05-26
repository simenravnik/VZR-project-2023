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
#include "../../lib/models/mlp_model.h"

#include "train_mlp_mpi.h"

int main(int argc, char** argv) {

    // Prepare dataset
    TrainTestSplit split = prepare_dataset(FILEPATH, CLASSES);

    printf("YET TO BE IMPLEMENTED\n");

    MLP_model model = train_mlp_mpi(split.X_train, split.Y_train, HIDDEN_SIZE, ETA, BATCH_SIZE, EPOCHS);

    // Test the model
    float accuracy = predict(split.X_train, split.Y_train, model);
    printf("Accuracy: %f\n\n", accuracy);

    // Free memory
    free_split(split);
    free_model(model);
    
    return 0;
}