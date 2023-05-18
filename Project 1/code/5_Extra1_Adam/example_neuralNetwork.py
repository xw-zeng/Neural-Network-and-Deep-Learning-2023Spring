import numpy as np
from scipy.io import loadmat
import math
from linearInd2Binary import linearInd2Binary
from MLPclassificationLoss import MLPclassificationLoss
from MLPclassificationPredict import MLPclassificationPredict
from standardizeCols import standardizeCols
import matplotlib.pyplot as plt
from adam import adam

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
inputFeatures = 200 # first layer
nHidden = []
prev_layer = d + 1 # take bias into account
inputWeights = np.random.randn(prev_layer, inputFeatures)
prev_layer = inputFeatures + 1
hiddenWeights = []
S = []
R = []
theta = []
for h in range(len(nHidden)):
    hiddenWeights.append(np.random.randn(prev_layer, nHidden[h]))
    S.append(np.zeros((prev_layer, nHidden[h])))
    R.append(np.zeros((prev_layer, nHidden[h])))
    theta.append(np.zeros((prev_layer, nHidden[h])))
    prev_layer = nHidden[h] + 1
outputWeights = np.random.randn(prev_layer, nLabels)
Weights = [inputWeights, hiddenWeights, outputWeights]
S = [np.zeros(inputWeights.shape), S, np.zeros(outputWeights.shape)]
R = [np.zeros(inputWeights.shape), R, np.zeros(outputWeights.shape)]
theta = [np.zeros(inputWeights.shape), theta, np.zeros(outputWeights.shape)]

# Train with stochastic gradient
maxIter = 20000
alpha = 0.99e-3
interval = 20
verr = np.zeros(interval)
trerr = np.zeros(interval)
for it in range(0, maxIter):
    if (it) % round(maxIter / interval) == 0:
        _, yhat, _, _ = MLPclassificationPredict(Weights, Xvalid)
        _, ytrain, _, _ = MLPclassificationPredict(Weights, X)
        verr[it // round(maxIter / interval)] = np.sum(yhat != (yvalid - 1)[:, 0]) / t
        trerr[it // round(maxIter / interval)] = np.sum(ytrain != (y - 1)[:, 0]) / n
        print('Training iteration = %d, validation error = %f'%(it, verr[it // round(maxIter / interval)]))
    i = math.ceil(np.random.uniform(0, n))
    _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :])
    Weights[2], theta[2], S[2], R[2] = adam(g[2], Weights[2], theta[2], S[2], R[2], it + 1, alpha)
    Weights[0], theta[0], S[0], R[0] = adam(g[0], Weights[0], theta[0], S[0], R[0], it + 1, alpha)
    for i in range(len(Weights[1])):
        Weights[1][i], theta[1][i], S[1][i], R[1][i] = adam(g[1][i], Weights[1][i], theta[1][i],
                                                            S[1][i], R[1][i], it + 1, alpha)

# Evaluate test error
_, yhat, _, _ = MLPclassificationPredict(Weights, Xtest)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
print('Test error with final model = %f'%te)

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
