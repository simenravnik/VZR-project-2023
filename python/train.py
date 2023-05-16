import numpy as np

# Function used to load iris data set
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=0)
    X = data[:, :4]  # input features
    y = data[:, 4]   # target labels
    num_classes = len(np.unique(y))
    # use one-hot encoding of the class
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y.astype(int)] = 1

    return X, y_one_hot

# Train the model
def train_mlp(X, Y, hiddenSize, eta, batchSize, epochs):
    samples, features = X.shape
    outputs = Y.shape[1]
    W1 = np.random.randn(features, hiddenSize)
    W2 = np.random.randn(hiddenSize, outputs)
    b1 = np.zeros(hiddenSize)
    b2 = np.zeros(outputs)
    for epoch in range(epochs):
        for i in range(0, samples, batchSize):
            Xb = X[i:i+batchSize]
            Yb = Y[i:i+batchSize]
            # forward phase
            H = np.tanh(np.dot(Xb, W1) + b1)
            Y_hat = np.tanh(np.dot(H, W2) + b2)

            # backpropagation
            E = Y_hat-Yb
            W2g = np.dot(H.T, E*(1-np.square(Y_hat)))
            b2g = np.sum(E*(1-np.square(Y_hat)), axis=0)
            He = np.dot(E*(1-np.square(Y_hat)), W2.T)*(1-np.square(H))
            W1g = np.dot(Xb.T, He)
            b1g = np.sum(He, axis=0)

            # adjust the weights of the model
            W1 = W1-eta*W1g
            b1 = b1-eta*b1g
            W2 = W2-eta*W2g
            b2 = b2-eta*b2g

    return (W1, b1, W2, b2)

# Test the accuracy of the model
def predict(X, Y, W1, b1, W2, b2):
    H = np.tanh(np.dot(X, W1) + b1)
    Y_hat = np.tanh(np.dot(H, W2) + b2)
    predicted_labels = np.argmax(Y_hat, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


# Example usage on random data
X = np.random.randn(100, 10)  # input data
Y = np.random.randint(0, 2, size=(100, 2))  # target labels (one-hot encoded)
W1, b1, W2, b2 = train_mlp(X, Y, 10, 0.1, 10, 100)
accuracy = predict(X, Y, W1, b1, W2, b2)
print("Accuracy of the model on random data: "+str(accuracy))
#------------------------------------------------------------------------------
# Example usage on iris data set
X, Y = load_data("../data/iris.data")
# Normalize the input features (optional but recommended)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Train
W1, b1, W2, b2 = train_mlp(X, Y, 20, 0.1, 10, 1000)
# Predict
accuracy = predict(X, Y, W1, b1, W2, b2)
print("Accuracy of the model on iris data set: "+str(accuracy))
