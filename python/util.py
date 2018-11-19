import numpy as np

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
