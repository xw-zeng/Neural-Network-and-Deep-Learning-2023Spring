import numpy as np
from MLPclassificationPredict import MLPclassificationPredict


def sech_square(x):
    return np.multiply(1 / np.cosh(x), 1 / np.cosh(x))


def MLPclassificationLoss(Weights, X, y):
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    nInstances, _ = X.shape
    inputWeights, hiddenWeights, outputWeights = Weights
    yhat, _, ip, fp = MLPclassificationPredict(Weights, X)
    bias = np.ones((nInstances, 1))

    relativeErr = yhat - y
    f = np.sum(relativeErr ** 2)
    err = 2 * relativeErr
    # Output Weights
    gOutput = np.dot(np.concatenate((bias, fp[-1]), axis = 1).T, err)
    err = np.multiply(sech_square(ip[-1]), np.dot(err, outputWeights[1:, :].T))
    # Hidden Weights
    gHidden = [0] * len(hiddenWeights)
    for h in range(len(hiddenWeights) - 1, -1, -1):
        gHidden[h] = np.dot(np.concatenate((bias, fp[h]), axis = 1).T, err)
        err = np.multiply(sech_square(ip[h]), np.dot(err, hiddenWeights[h][1:, :].T))
    # Input Weights
    gInput = np.dot(np.concatenate((np.ones((nInstances, 1)), X), axis = 1).T, err)
    g = [gInput, gHidden, gOutput]
    return f, g
