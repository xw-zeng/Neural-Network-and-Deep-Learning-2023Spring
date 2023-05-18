import numpy as np
from scipy.io import loadmat
import math
from linearInd2Binary import linearInd2Binary
from MLPclassificationLoss import MLPclassificationLoss
from MLPclassificationPredict import MLPclassificationPredict
from standardizeCols import standardizeCols
import matplotlib.pyplot as plt

m = loadmat('../data/digits.mat')
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
X = np.concatenate((np.ones((n, 1)), X), axis=1)
d += 1

# Make sure to apply the same transformation to the validation/test data
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xvalid = np.concatenate((np.ones((t, 1)), Xvalid), axis=1)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
Xtest = np.concatenate((np.ones((t2, 1)), Xtest), axis=1)


def Q1(inputFeatures):
    # Form Weights
    inputFeatures = inputFeatures # first layer
    nHidden = []
    prev_layer = d + 1 # take bias into account
    inputWeights = np.random.randn(prev_layer, inputFeatures)
    prev_layer = inputFeatures
    hiddenWeights = []
    for h in range(len(nHidden)):
        hiddenWeights.append(np.random.randn(prev_layer, nHidden[h]))
        prev_layer = nHidden[h]
    outputWeights = np.random.randn(prev_layer, nLabels)
    Weights = [inputWeights, hiddenWeights, outputWeights]

    # Train with stochastic gradient
    verr = np.zeros(interval)
    trerr = np.zeros(interval)
    for it in range(0, maxIter):
        if (it) % round(maxIter / interval) == 0:
            _, yhat, _, _ = MLPclassificationPredict(Weights, Xvalid)
            _, ytrain, _, _ = MLPclassificationPredict(Weights, X)
            verr[it // round(maxIter / interval)] = np.sum(yhat != (yvalid - 1)[:, 0]) / t
            trerr[it // round(maxIter / interval)] = np.sum(ytrain != (y - 1)[:, 0]) / n
        i = math.ceil(np.random.uniform(0, n))
        _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :])
        Weights[2] -= alpha * g[2]
        Weights[0] -= alpha * g[0]
        if len(Weights[1]) != 0:
            Weights[1] = [Weights[1][h] - alpha * g[1][h] for h in range(len(Weights[1]))]

    # Evaluate test error
    _, yhat, _, _ = MLPclassificationPredict(Weights, Xtest)
    te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
    return verr, trerr, te


maxIter = 20000
alpha = 1e-3
interval = 20
v = []
tr = []
te = []
for i in [10, 20, 50, 100, 200, 400]:
    verr, trerr, terr = Q1(i)
    v.append(verr)
    tr.append(trerr)
    te.append(terr)

plt.figure(figsize = (16, 10))
plt.rcParams['figure.dpi'] = 150
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(range(0, maxIter, round(maxIter / interval)), v[i], label = 'Validation')
    plt.plot(range(0, maxIter, round(maxIter / interval)), tr[i], label = 'Training')
    plt.legend()
    plt.title(f'SSE loss: Test error = {te[i]:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
plt.show()
