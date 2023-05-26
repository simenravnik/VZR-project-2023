#ifndef PARAMETERS_H
#define PARAMETERS_H

const char* FILEPATH = "data/winequalityN.csv";
const float TRAIN_SIZE_PERCENTAGE = 1.0;    // If 1.0, we use the training set for training and testing
const int CLASSES = 10;

const int HIDDEN_SIZE = 50;
const int BATCH_SIZE = 10;
const int EPOCHS = 100;
const float ETA = 0.01;

#endif