#ifndef MODEL_H
#define MODEL_H

#include "../matrix/matrix.h"

typedef struct MLP_model {
    Matrix W1;
    Matrix W2;
    Matrix b1;
    Matrix b2;
} MLP_model;

#endif
