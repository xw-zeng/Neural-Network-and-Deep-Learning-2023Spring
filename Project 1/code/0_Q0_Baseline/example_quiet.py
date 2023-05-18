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
X = np.concatenate((np.ones((n, 1)), X), axis=1)
d += 1

# Make sure to apply the same transformation to the validation/test data
Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
Xvalid = np.concatenate((np.ones((t, 1)), Xvalid), axis=1)
Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
Xtest = np.concatenate((np.ones((t2, 1)), Xtest), axis=1)

# Choose network structure
nHidden = [10]

# Count number of parameters and initialize weights 'w'
nParams = d * nHidden[0]
for h in range(1, len(nHidden)):
    nParams += nHidden[h - 1] * nHidden[h]
nParams += nHidden[len(nHidden) - 1] * nLabels
w = np.random.randn(nParams, 1)

# Train with stochastic gradient
maxIter = 100000
alpha = 1e-3
for it in range(0, maxIter):
    i = math.ceil(np.random.uniform(0, n))
    _, g = MLPclassificationLoss(w, X[i - 1, :], yExpanded[i - 1, :], nHidden, nLabels)
    w = w - alpha * g

# Evaluate test error
yhat = MLPclassificationPredict(w, Xtest, nHidden, nLabels)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
