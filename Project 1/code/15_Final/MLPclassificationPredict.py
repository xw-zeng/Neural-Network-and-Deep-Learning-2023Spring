import numpy as np


def MLPclassificationPredict(Weights, X):
    X = np.atleast_2d(X)
    nInstances, _ = X.shape
    inputWeights, hiddenWeights, outputWeights = Weights
    bias = np.ones((nInstances, 1))

    # Compute Output: Matrix Operations
    ip = [np.dot(np.concatenate((bias, X), axis=1), inputWeights)]
    fp = [(ip[0] > 0) * ip[0]]
    for h in range(1, len(hiddenWeights) + 1):
        ip.append(np.dot(np.concatenate((bias, fp[h - 1]), axis=1), hiddenWeights[h - 1]))
        fp.append((ip[h] > 0) * ip[h])
    yhat = np.dot(np.concatenate((bias, fp[-1]), axis=1), outputWeights)
    y = np.argmax(yhat, axis=1)
    return yhat, y, ip, fp