import numpy as np


def linearInd2Binary(ind, nLabels):
    """Transform the labels into {-1, 1} encoding."""
    n = len(ind)
    y = - np.ones((n, nLabels))
    for i in range(0, n):
        y[i, ind[i, 0] - 1] = 1
    return y
