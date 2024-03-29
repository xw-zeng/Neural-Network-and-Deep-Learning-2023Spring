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

# Make sure to apply the same transformation to the validation/test data
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)

# Form Weights
inputFeatures = 10 # first layer
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
maxIter = 100000
alpha = 1e-3
interval = 20
verr = np.zeros(interval)
trerr = np.zeros(interval)
for it in range(0, maxIter):
    if (it) % round(maxIter / interval) == 0:
        _, yhat, _, _ = MLPclassificationPredict(Weights, Xvalid)
        _, ytrain, _, _ = MLPclassificationPredict(Weights, X)
        verr[it // round(maxIter / interval)] = np.sum(yhat != (yvalid - 1)[:, 0]) / t
        trerr[it // round(maxIter / interval)] = np.sum(ytrain != (y - 1)[:, 0]) / n
        print('Training iteration = %d, validation error = %f' % (it, verr[it // round(maxIter / interval)]))
    i = math.ceil(np.random.uniform(0, n))
    _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :])
    Weights[2] -= alpha * g[2]
    Weights[0] -= alpha * g[0]
    if len(Weights[1]) != 0:
        Weights[1] = [Weights[1][h] - alpha * g[1][h] for h in range(len(Weights[1]))]

# Evaluate test error
_, yhat, _, _ = MLPclassificationPredict(Weights, Xtest)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
print('Test error with final model = %f' % te)

# Plot the error
plt.figure(figsize=(8, 6))
plt.rcParams['figure.dpi'] = 150
plt.plot(range(0, maxIter, round(maxIter / interval)), verr, label='Validation')
plt.plot(range(0, maxIter, round(maxIter / interval)), trerr, label='Training')
plt.legend()
plt.title(f'SSE loss: Test error = {te:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
