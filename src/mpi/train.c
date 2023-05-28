/*
 * Search for "TODO" comments to find the parts that need improvements.
*/

#include "/usr/include/openmpi-x86_64/mpi.h"
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

#include "train_mpi.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prepare dataset
    TrainTestSplit split = prepare_dataset(FILEPATH, CLASSES, TRAIN_SIZE_PERCENTAGE);

    printf("YET TO BE IMPLEMENTED\n");

    MLP_model model = train_mpi(split.X_train, split.Y_train, HIDDEN_SIZE, ETA, BATCH_SIZE, EPOCHS, rank, num_procs);

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

    MPI_Finalize();
    return 0;
}