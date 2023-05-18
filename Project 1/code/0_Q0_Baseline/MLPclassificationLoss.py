import numpy as np


def sech_square(x):
    return np.multiply(1 / np.cosh(x), 1 / np.cosh(x))


def MLPclassificationLoss(w, X, y, nHidden, nLabels):
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
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

    f = 0
    gInput = np.zeros((inputWeights.shape))
    gHidden = []
    for h in range(1, len(nHidden)):
        gHidden.append(np.zeros((hiddenWeights[h - 1].shape)))
    gOutput = np.zeros((outputWeights.shape))

    # Compute Output
    yhat = np.zeros((nInstances, outputWeights.shape[1]))
    for i in range(0, nInstances):
        ip = [np.dot(np.atleast_2d(X[i, :]), inputWeights)]
        fp = [np.tanh(ip[0])]
        for h in range(1, len(nHidden)):
            ip.append(np.dot(fp[h - 1], hiddenWeights[h - 1]))
            fp.append(np.tanh(ip[h]))
        yhat[i, :] = np.dot(fp[-1], outputWeights)

        relativeErr = yhat[i, :] - y[i, :]
        f += np.sum(relativeErr ** 2)
        err = 2 * relativeErr

        # Output Weights
        for c in range(0, nLabels):
            gOutput[:, c] = gOutput[:, c] + err[c] * fp[-1]

        if len(nHidden) > 1:
            # Last Layer of Hidden Weights
            backprop = np.zeros((nLabels, outputWeights.shape[0]))
            for c in range(0, nLabels):
                backprop[c, :] = err[0, c] * (np.multiply(sech_square(ip[-1]), outputWeights[:, c].T))
                gHidden[-1] += np.dot(fp[-2].T, np.atleast_2d(backprop[c, :]))
            backprop = np.sum(backprop, axis=1)

            # Other Hidden Layers
            for h in range(len(nHidden) - 3, -1, -1):
                backprop = np.multiply(np.dot(np.atleast_2d(backprop), hiddenWeights[h + 1].T
                                              ), sech_square(ip[h + 1]))
                gHidden[h] += np.dot(fp[h].T, backprop)

            # Input Weights
            backprop = np.multiply(np.dot(np.atleast_2d(backprop), hiddenWeights[0].T), sech_square(ip[0]))
            gInput += np.dot(np.atleast_2d(X[i, :]).T, backprop)
        else:
            # Input Weights
            for c in range(0, nLabels):
                gInput += np.dot(np.atleast_2d(err[c] * X[i, :]).T,
                                 np.multiply(sech_square(ip[-1]), outputWeights[:, c].T))

    # Put Gradient into vector
    g = np.zeros((w.shape))
    g[0:(nVars * nHidden[0])] = gInput.reshape((nVars * nHidden[0], 1), order='F')
    offset = nVars * nHidden[0]
    for h in range(1, len(nHidden)):
        g[offset:(offset + nHidden[h - 1] * nHidden[h])] = gHidden[h - 1].reshape(
            (nHidden[h - 1] * nHidden[h], 1), order='F')
        offset += nHidden[h - 1] * nHidden[h]
    g[offset:(offset + nHidden[-1] * nLabels)] = gOutput.reshape((nHidden[-1] * nLabels, 1), order='F')
    return f, g
