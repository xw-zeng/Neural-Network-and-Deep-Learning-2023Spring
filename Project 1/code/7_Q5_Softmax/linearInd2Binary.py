import numpy as np


def linearInd2Binary(ind, nLabels):
    """Transform the labels into {0, 1} encoding adaptive to softmax function."""
    n = len(ind)
    y = np.zeros((n, nLabels))
    for i in range(0, n):
        y[i, ind[i, 0] - 1] = 1
    return y
