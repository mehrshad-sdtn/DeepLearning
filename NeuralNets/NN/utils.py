import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def get_activation(name):
    if name == "sigmoid": return sigmoid
    if name == "relu": return relu
