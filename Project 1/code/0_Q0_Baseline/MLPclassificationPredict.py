import numpy as np


def MLPclassificationPredict(w, X, nHidden, nLabels):
    X = np.atleast_2d(X)
    nInstances, nVars = X.shape

    # Form Weights
    inputWeights = w[0:(nVars * nHidden[0])].reshape((nVars, nHidden[0]), order='F')  # By column
    offset = nVars * nHidden[0]
    hiddenWeights = []
    for h in range(1, len(nHidden)):
        hiddenWeights.append(w[offset:offset + nHidden[h - 1] * nHidden[h]
                             ].reshape((nHidden[h - 1], nHidden[h]), order='F'))
        offset += nHidden[h - 1] * nHidden[h]
    outputWeights = w[offset:(offset + nHidden[-1] * nLabels)]
    outputWeights = outputWeights.reshape((nHidden[-1], nLabels), order='F')

    # Compute Output
    yhat = np.zeros((nInstances, outputWeights.shape[1]))
    for i in range(0, nInstances):
        ip = [np.dot(np.atleast_2d(X[i, :]), inputWeights)]
        fp = [np.tanh(ip[0])]
        for h in range(1, len(nHidden)):
            ip.append(np.dot(fp[h - 1], hiddenWeights[h - 1]))
            fp.append(np.tanh(ip[h]))
        yhat[i, :] = np.dot(fp[-1], outputWeights)
    y = np.argmax(yhat, axis=1)
    return y
