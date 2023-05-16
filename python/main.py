import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_mlp(X, Y, hiddenSize, alpha, batchSize, epochs):
    samples, features = X.shape
    samples, outputs = Y.shape
    W1 = np.random.randn(features, hiddenSize)     # First layer weights
    b1 = np.random.randn(hiddenSize)               # First layer biases
    W2 = np.random.randn(hiddenSize, outputs)      # Second layer weights
    b2 = np.random.randn(outputs)                  # Second layer biases 
    batches = int(np.ceil(samples / batchSize))
    
    for epoch in range(epochs):
        for b in range(batches):
            X_b = X[b:b+batchSize]                       # Input data for the current batch
            Y_b = Y[b:b+batchSize]                       # Target data for the current batch
            
            H = np.tanh(X_b.dot(W1) + b1)                # Compute the hidden layer activations
            Y_hat = np.tanh(H.dot(W2) + b2)              # Compute the network output
            
            E = Y_hat - Y_b                              # Error between the predicted and target outputs

            W2_g = H.T.dot(E * (1 - Y_hat**2))           # Gradient of the second layer weights
            b2_g = np.sum(E * (1 - Y_hat**2), axis=0)    # Gradient of the second layer biases
            
            H_e = (E * (1 - Y_hat**2)).dot(W2.T)         # Compute the error propagated to the hidden layer
            H_e = H_e * (1 - H**2)                       # Apply the derivative of the activation function to the hidden layer error
            
            W1_g = X_b.T.dot(H_e)                        # Compute the gradient of the first layer weights
            b1_g = np.sum(H_e, axis=0)                   # Compute the gradient of the first layer biases
            
            W1 = W1 - alpha * W1_g                       # Update the first layer weights
            b1 = b1 - alpha * b1_g                       # Update the first layer biases
            W2 = W2 - alpha * W2_g                       # Update the second layer weights
            b2 = b2 - alpha * b2_g                       # Update the second layer biases
            
    return W1, b1, W2, b2

if __name__ == "__main__":
    
    data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'iris.csv'))
    # data = data.sample(frac=1)  # shuffle the data

    print(data.head())

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    Y = Y.reshape(-1, 1)

    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    hiddenSize = 10
    alpha = 0.001
    batchSize = 5
    epochs = 1000
    
    W1, b1, W2, b2 = train_mlp(X_train, Y_train, hiddenSize, alpha, batchSize, epochs)

    # Compute the predicted output for the test set
    H = np.tanh(np.dot(X_test, W1) + b1)
    Y_pred = np.tanh(np.dot(H, W2) + b2)

    print("Majority class =", 1 - (np.count_nonzero(Y_test) / Y_test.size))

    # Compute the predicted class labels for the test set
    Y_pred_labels = np.where(Y_pred > 0.5, 1, 0)

    # Compute the accuracy of the model on the test set
    accuracy = np.mean(Y_pred_labels == Y_test.flatten())
    print("Accuracy:", accuracy)