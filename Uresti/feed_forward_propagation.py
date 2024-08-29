import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(X, weights1, weights2, bias1, bias2):
    # Input layer to hidden layer
    z1 = np.dot(X, weights1) + bias1
    a1 = sigmoid(z1)
    
    # Hidden layer to output layer
    z2 = np.dot(a1, weights2) + bias2
    a2 = sigmoid(z2)
    
    return a2

# Example usage:
# X is the input data, with one example per row
X = np.array([[0.1, 0.2, 0.7], 
              [0.5, 0.6, 0.1]])

# Weights for the input layer to hidden layer (3 inputs, 4 neurons in hidden layer)
weights1 = np.array([[0.1, 0.2, 0.3, 0.4], 
                     [0.5, 0.6, 0.7, 0.8], 
                     [0.9, 1.0, 1.1, 1.2]])

# Weights for the hidden layer to output layer (4 neurons in hidden layer, 2 outputs)
weights2 = np.array([[0.1, 0.2], 
                     [0.3, 0.4], 
                     [0.5, 0.6], 
                     [0.7, 0.8]])

# Biases for each layer
bias1 = np.array([0.1, 0.2, 0.3, 0.4])
bias2 = np.array([0.1, 0.2])

# Perform feedforward propagation
output = feedforward(X, weights1, weights2, bias1, bias2)
print("Output:", output)
