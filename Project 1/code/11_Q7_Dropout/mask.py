import numpy as np


def mask(X, p, train=True):
    _, nVars = X.shape
    if train:
        return (np.random.randn(nVars) > p) * X / (1 - p)
    else:
        return X
