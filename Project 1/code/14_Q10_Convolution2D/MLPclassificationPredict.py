import numpy as np
from scipy.signal import convolve2d as conv2


def MLPclassificationPredict(Weights, X, new_size):
    X = np.atleast_2d(X)
    nInstances, nVars = X.shape
    convWeights, inputWeights, hiddenWeights, outputWeights = Weights
    nChannels = len(convWeights) - 1  # 通道数量
    bias = np.ones((nInstances, 1))  # 构造一个bias

    # Compute Output: Matrix Operations
    C = np.zeros((nInstances, nChannels, new_size))  # 实例数、通道数、新的大小
    for i in range(nInstances):  # 遍历每一个实例
        for j in range(nChannels):  # 遍历每一个通道
            img = X[i, :].reshape((int(np.sqrt(nVars)), -1), order='F')  # 实例16*16矩阵
            conv = conv2(img, convWeights[j], 'valid') + convWeights[nChannels][j]  # 生成12*12矩阵并且为每个元素增加bias
            C[i, j] = conv.reshape((-1,), order='F')  # 将新生成矩阵的元素向量化装入C
    ip = [np.dot(np.concatenate((bias, C.reshape(nInstances, -1)), axis=1), inputWeights)]
    fp = [(ip[0] > 0) * ip[0]]  # 使用RELU激活函数
    for h in range(len(hiddenWeights)):
        ip.append(np.dot(np.concatenate((bias, fp[h]), axis=1), hiddenWeights[h]))
        fp.append((ip[h + 1] > 0) * ip[h + 1])
    yhat = np.dot(np.concatenate((bias, fp[-1]), axis=1), outputWeights)
    y = np.argmax(yhat, axis=1)
    return yhat, y, ip, fp, C
