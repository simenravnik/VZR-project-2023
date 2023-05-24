#ifndef TRAIN_MLP_SERIAL_H
#define TRAIN_MLP_SERIAL_H

#include "../../lib/matrix/matrix.h"
#include "../../lib/models/mlp_model.h"

MLP_model train_mlp_serial(Matrix X, Matrix Y, int hiddenSize, float eta, int batchSize, int epochs);

#endif
