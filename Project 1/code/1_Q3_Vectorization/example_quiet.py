import numpy as np
from scipy.io import loadmat
import math
from linearInd2Binary import linearInd2Binary
from MLPclassificationLoss import MLPclassificationLoss
from MLPclassificationPredict import MLPclassificationPredict
from standardizeCols import standardizeCols

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
for it in range(0, maxIter):
    i = math.ceil(np.random.uniform(0, n))
    _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :])
    Weights[2] -= alpha * g[2]
    Weights[0] -= alpha * g[0]
    if len(Weights[1]) != 0:
        Weights[1] = [Weights[1][h] - alpha * g[1][h] for h in range(len(Weights[1]))]

# Evaluate test error
_, yhat, _, _ = MLPclassificationPredict(Weights, Xtest)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
