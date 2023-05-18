import numpy as np
from MLPclassificationPredict import MLPclassificationPredict
from fine_tune import fine_tune


def MLPclassificationLoss(Weights, X, y, penalty):
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    nInstances, _ = X.shape
    inputWeights, hiddenWeights, outputWeights = Weights
    yhat, _, ip, fp = MLPclassificationPredict(Weights, X)
    bias = np.ones((nInstances, 1))

    yhat = yhat.T - np.max(yhat.T, axis=0)
    s = np.exp(yhat)
    yhat = (s / np.sum(s, axis=0)).T
    f = fine_tune(yhat, y, outputWeights, penalty)

    # Output Weights
    gOutput = np.dot(np.concatenate((bias, fp[-1]), axis=1).T, yhat - y) + penalty * outputWeights
    err = yhat - y
    err = np.multiply((ip[-1] > 0), np.dot(err, outputWeights[1:, :].T))
    # Hidden Weights
    gHidden = [0] * len(hiddenWeights)
    for h in range(len(hiddenWeights) - 1, -1, -1):
        gHidden[h] = np.dot(np.concatenate((bias, fp[h]), axis=1).T, err) + penalty * hiddenWeights[h]
        err = np.multiply((ip[h] > 0), np.dot(err, hiddenWeights[h][1:, :].T))
    # Input Weights
    gInput = np.dot(np.concatenate((bias, X), axis=1).T, err) + penalty * inputWeights
    g = [gInput, gHidden, gOutput]
    return f, g
