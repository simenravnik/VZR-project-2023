#ifndef PARAMETERS_H
#define PARAMETERS_H

const char* FILEPATH = "data/mnist.csv";
const float TRAIN_SIZE_PERCENTAGE = 0.8;    // If 1.0, we use the training set for training and testing
const int CLASSES = 10;

const int HIDDEN_SIZE = 100;
const int BATCH_SIZE = 20;
const int EPOCHS = 10;
const float ETA = 0.01;

// CUDA
const int CUDA_BLOCK_SIZE = 32;

#endif