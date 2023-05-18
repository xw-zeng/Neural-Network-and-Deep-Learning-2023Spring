import numpy as np
from MLPclassificationPredict import MLPclassificationPredict
from fine_tune import fine_tune
from scipy.signal import convolve2d as conv2


def MLPclassificationLoss(Weights, X, y, new_size, penalty):
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    nInstances, nVars = X.shape
    convWeights, inputWeights, hiddenWeights, outputWeights = Weights
    nChannels = len(convWeights) - 1
    yhat, _, ip, fp, C = MLPclassificationPredict(Weights, X, new_size)
    bias = np.ones((nInstances, 1))

    yhat = yhat.T - np.max(yhat.T, axis=0)
    s = np.exp(yhat)
    yhat = (s / np.sum(s, axis=0)).T
    f = fine_tune(yhat, y, outputWeights, penalty)
    err = yhat - y

    # Output Weights
    gOutput = (np.dot(np.concatenate((bias, fp[-1]), axis=1).T, err) + penalty * outputWeights) / nInstances
    err = np.multiply((ip[-1] > 0), np.dot(err, outputWeights[1:, :].T))
    # Hidden Weights
    gHidden = [0] * len(hiddenWeights)
    for h in range(len(hiddenWeights) - 1, -1, -1):
        gHidden[h] = (np.dot(np.concatenate((bias, fp[h]), axis=1).T, err) + penalty * hiddenWeights[h]) / nInstances
        err = np.multiply((ip[h] > 0), np.dot(err, hiddenWeights[h][1:, :].T))
    # err = np.dot(((np.identity(nInstances) - 1 / nInstances * np.ones((nInstances, nInstances)))).T, err)
    # Input Weights
    gInput = (np.dot(np.concatenate((bias, C.reshape(nInstances, -1)), axis=1).T,
                     err) + penalty * inputWeights) / nInstances
    err = np.dot(err, inputWeights[1:, :].T)
    # Convolutional Weights
    gConv = []
    for j in range(nChannels + 1):
        gConv.append(penalty * convWeights[j])  # 正则化
    err = err.reshape((nInstances, nChannels, new_size))
    gConv[nChannels] = (gConv[nChannels] + np.sum(err, axis=(0, 2)).reshape(-1, 1)) / nInstances
    for j in range(nChannels):  # 遍历每个通道
        for i in range(nInstances):  # 遍历每一个实例
            reverseX = X[i, :].reshape((int(np.sqrt(nVars)), -1), order='F')  # rot180(X)
            gConv[j] = gConv[j] + conv2(reverseX, err[i, j, :].reshape((int(np.sqrt(new_size)), -1), order='F'),
                                        'valid')
        gConv[j] = gConv[j] / nInstances
    g = [gConv, gInput, gHidden, gOutput]
    return f, g
