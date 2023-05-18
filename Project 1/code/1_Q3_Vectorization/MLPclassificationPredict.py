import numpy as np


def MLPclassificationPredict(Weights, X):
    X = np.atleast_2d(X)
    nInstances, _ = X.shape
    inputWeights, hiddenWeights, outputWeights = Weights

    # Compute Output: Matrix Operations
    ip = [np.dot(np.concatenate((np.ones((nInstances, 1)), X), axis=1), inputWeights)]
    fp = [np.tanh(ip[0])]
    for h in range(1, len(hiddenWeights) + 1):
        ip.append(np.dot(fp[h - 1], hiddenWeights[h - 1]))
        fp.append(np.tanh(ip[h]))
    yhat = np.dot(fp[-1], outputWeights)
    y = np.argmax(yhat, axis=1)
    return yhat, y, ip, fp
