# Acceleration of multilayer perceptron training

> A Comparative Study of Serial, CUDA, MPI, and OpenMP Implementations

## Compile and run

```
./run_nsc.sh
or
./run_arnes.sh
```

## Project structure

```
.
├── README.md
├── data
├── lib
│   ├── helpers
│   ├── matrix
│   │   ├── matrix.h
│   │   ├── matrix_cuda.h
│   │   ├── matrix_cuda_sk.h
│   │   ├── matrix_mpi.h
│   │   ├── matrix_openmp.h
│   │   └── matrix_serial.h
│   ├── models
│   │   └── mlp_model.h
│   ├── read
│   │   └── read.h
│   └── time
│       └── cuda_timer.h
├── parameters.h
├── results
├── run_arnes.sh
├── run_nsc.sh
└── src
    ├── cuda
    │   ├── test
    │   ├── train.cu
    │   ├── train_cuda.h
    │   └── train_cuda_sk.h
    ├── mpi
    │   ├── test
    │   ├── train.c
    │   └── train_mpi.h
    ├── openmp
    │   ├── train.c
    │   └── train_openmp.h
    └── serial
        ├── train.c
        └── train_serial.h

```


## Pseudocode
Training of a double layer perceptron using back-propagation.

```bash
function train_mlp(X, Y, hiddenSize, alpha, batchSize, epochs)
  samples, features ← size of X
  samples, outputs ← size of Y
  W1 ← random matrix of size (features, hiddenSize) # First layer weights
  b1 ← random vector of size hiddenSize             # First layer biases
  W2 ← random matrix of size (hiddenSize, outputs)  # Second layer weights
  b2 ← random vector of size outputs                # Second layer biases 
  batches ← ceil(samples / batchSize)
  for epoch in 1 to epochs do
    for b in 1 to batches do
      batch_rows ← rows of batch b                  # Get the row indices for the current batch

      X_b ← X[batch_rows, :]                        # Input data for the current batch
      Y_b ← Y[batch_rows, :]                        # Target data for the current batch

      H ← tanh(X_b * W1 + b1)                       # Compute the hidden layer activations
      Y_hat ← tanh(H * W2 + b2)                     # Compute the network output

      E ← Y_hat − Y_b                               # Error between the predicted and target outputs

      # ◦ is hamarad product (element-wise)
      # ^(◦2) is element-wise squaring
      W2_g ← H_transposed * (E ◦ (1 − Y_hat^(◦2)))  # Gradient of the second layer weights
      b2_g ← 1_transposed * (E ◦ (1 − Y_hat^(◦2)))  # Gradient of the second layer biases

      H_e ← (E ◦ (1 − Y_hat^(◦2))) * W2_transposed  # Compute the error propagated to the hidden layer
      H_e ← H_e ◦ (1 − H^(◦2))                      # Apply the derivative of the activation function to the hidden layer error

      W1_g ← X_b_transposed * H_e                   # Compute the gradient of the first layer weights
      b1_g ← 1_transposed * H_e                     # Compute the gradient of the first layer biases

      W1 ← W1 − alpha * W1_g                        # Update the first layer weights
      b1 ← b1 − alpha * b1_g                        # Update the first layer biases
      W2 ← W2 − alpha * W2_g                        # Update the second layer weights
      b2 ← b2 − alpha * b2_g                        # Update the second layer biases
    end for
  end for
  return W1, b1,W2, b2
end function
```

## Results

![Results](results/results.png)