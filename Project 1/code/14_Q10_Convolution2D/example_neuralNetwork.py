import numpy as np
from scipy.io import loadmat
import math
from linearInd2Binary import linearInd2Binary
from MLPclassificationLoss import MLPclassificationLoss
from MLPclassificationPredict import MLPclassificationPredict
from standardizeCols import standardizeCols
import matplotlib.pyplot as plt
from momentum import momentum

m = loadmat('../data/newdigits.mat')
X = m['X']
y = m['y']
Xvalid = m['Xvalid']
Xtest = m['Xtest']
yvalid = m['yvalid']
ytest = m['ytest']

n, d = X.shape
nLabels = y.max()
yExpanded = linearInd2Binary(y, nLabels)
t = Xvalid.shape[0]
t2 = Xtest.shape[0]

# Standardize columns and add bias
X, mu, sigma = standardizeCols(X)

# Make sure to apply the same transformation to the validation/test data
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)

# Form Weights
kernelsize = 3
nChannels = 3  # 卷积通道的数量
inputFeatures = 128  # first layer
nHidden = [32]
# Convolutional Weights
convWeights = []
for i in range(nChannels):  # 为每个卷积通道初始化权重
    convWeights.append(np.random.randn(kernelsize, kernelsize) * np.sqrt(2 / kernelsize ** 2))
convWeights.append(np.random.randn(nChannels, 1) * np.sqrt(2 / kernelsize ** 2))  # 每个卷积通道的bias
new_size = int((np.sqrt(d) - kernelsize + 1) ** 2)  # 计算卷积之后得到的新矩阵大小，在这里pad=0，stride=1
prev_layer = new_size * nChannels + 1  # 新矩阵的元素个数*通道数
# Input Weights
inputWeights = np.random.randn(prev_layer, inputFeatures) * np.sqrt(2 / prev_layer)
prev_layer = inputFeatures + 1
# Hidden Weights
hiddenWeights = []
for h in range(len(nHidden)):
    hiddenWeights.append(np.random.randn(prev_layer, nHidden[h]) * np.sqrt(2 / prev_layer))
    prev_layer = nHidden[h] + 1
# Output Weights
outputWeights = np.random.randn(prev_layer, nLabels) * np.sqrt(2 / prev_layer)
Weights = [convWeights, inputWeights, hiddenWeights, outputWeights]

# Train with stochastic gradient
maxIter = 50000
alpha = 2e-2
beta = 1
penalty = 1e-3
minibatch = 100
best = 1
interval = 50
prev_Weights = None
verr = np.zeros(interval)
trerr = np.zeros(interval)
for it in range(0, maxIter):
    if it == 0 or (it + 1) % round(maxIter / interval) == 0:
        _, yhat, _, _, _ = MLPclassificationPredict(Weights, Xvalid, new_size)
        _, ytrain, _, _, _ = MLPclassificationPredict(Weights, X, new_size)
        # Early Stopping
        if np.sum(yhat != (yvalid - 1)[:, 0]) / t - best <= 5e-4:
            Wbest = Weights[:]
            best = np.sum(yhat != (yvalid - 1)[:, 0]) / t
            verr[it // round(maxIter / interval)] = np.sum(yhat != (yvalid - 1)[:, 0]) / t
            trerr[it // round(maxIter / interval)] = np.sum(ytrain != (y - 1)[:, 0]) / n
        else:
            Weights = Wbest[:]
            verr[it // round(maxIter / interval)] = verr[it // round(maxIter / interval) - 1]
            trerr[it // round(maxIter / interval)] = trerr[it // round(maxIter / interval) - 1]
        print('Training iteration = %d, validation error = %f' % (it + 1, verr[it // round(maxIter / interval)]))

    i = np.ceil(np.random.uniform(0, n, minibatch)).astype(int)
    _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :], new_size, penalty)
    Weights = momentum(g, Weights, prev_Weights, alpha, beta, it)
    prev_Weights = Weights[:]

# Evaluate test error before fine tuning
_, yhat, _, _, _ = MLPclassificationPredict(Wbest, Xtest, new_size)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
print('Test error before fine tuning = %f' % te)

# Plot the error
plt.figure(figsize=(8, 6))
plt.rcParams['figure.dpi'] = 150
plt.plot(range(round(maxIter / interval), maxIter + round(maxIter / interval), round(maxIter / interval)), verr,
         label='Validation')
plt.plot(range(round(maxIter / interval), maxIter + round(maxIter / interval), round(maxIter / interval)), trerr,
         label='Training')
plt.legend()
plt.title(f'Softmax: Test error = {te:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

plt.figure(figsize=(8, 6))
plt.rcParams['figure.dpi'] = 150
plt.plot(range(round(maxIter / interval), maxIter + round(maxIter / interval), round(maxIter / interval)), verr,
         label='Validation')
plt.legend()
plt.title(f'Softmax: Test error = {te:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
