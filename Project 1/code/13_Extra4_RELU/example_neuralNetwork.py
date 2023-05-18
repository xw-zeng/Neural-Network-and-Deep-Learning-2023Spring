import numpy as np
from scipy.io import loadmat
import math
from linearInd2Binary import linearInd2Binary
from MLPclassificationLoss import MLPclassificationLoss
from MLPclassificationPredict import MLPclassificationPredict
from standardizeCols import standardizeCols
import matplotlib.pyplot as plt
from adam import adam
from fine_tune import fine_tune

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
inputFeatures = 256  # first layer
nHidden = [64]
prev_layer = d + 1  # take bias into account
inputWeights = np.random.randn(prev_layer, inputFeatures) * np.sqrt(2 / (prev_layer + inputFeatures))
prev_layer = inputFeatures + 1
hiddenWeights = []
S = []
R = []
theta = []
for h in range(len(nHidden)):
    hiddenWeights.append(np.random.randn(prev_layer, nHidden[h]) * np.sqrt(2 / (prev_layer + nHidden[h])))
    S.append(np.zeros((prev_layer, nHidden[h])))
    R.append(np.zeros((prev_layer, nHidden[h])))
    theta.append(np.zeros((prev_layer, nHidden[h])))
    prev_layer = nHidden[h] + 1
outputWeights = np.random.randn(prev_layer, nLabels) * np.sqrt(2 / (prev_layer + nLabels))
Weights = [inputWeights, hiddenWeights, outputWeights]
S = [np.zeros(inputWeights.shape), S, np.zeros(outputWeights.shape)]
R = [np.zeros(inputWeights.shape), R, np.zeros(outputWeights.shape)]
theta = [np.zeros(inputWeights.shape), theta, np.zeros(outputWeights.shape)]

# Train with stochastic gradient
maxIter = 20000
alpha = 1e-4
penalty = 1e-2
minibatch = 150
best = 1
interval = 20
verr = np.zeros(interval)
trerr = np.zeros(interval)
for it in range(0, maxIter):
    if it == 0 or (it + 1) % round(maxIter / interval) == 0:
        _, yhat, _, _ = MLPclassificationPredict(Weights, Xvalid)
        _, ytrain, _, _ = MLPclassificationPredict(Weights, X)
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
    _, g = MLPclassificationLoss(Weights, X[i - 1, :], yExpanded[i - 1, :], penalty)
    Weights[2], theta[2], S[2], R[2] = adam(g[2], Weights[2], theta[2], S[2], R[2], it + 1, alpha)
    Weights[0], theta[0], S[0], R[0] = adam(g[0], Weights[0], theta[0], S[0], R[0], it + 1, alpha)
    for i in range(len(Weights[1])):
        Weights[1][i], theta[1][i], S[1][i], R[1][i] = adam(g[1][i], Weights[1][i], theta[1][i],
                                                            S[1][i], R[1][i], it + 1, alpha)

# Evaluate test error before fine tuning
_, yhat, _, _ = MLPclassificationPredict(Wbest, Xtest)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
print('Test error before fine tuning = %f' % te)

m1 = loadmat('../data/digits.mat')
X_old = m1['X']
y_old = m1['y']
yExpanded_old = linearInd2Binary(y_old, nLabels)

# Fine Tuning with overall train data
Weights = Wbest[:]
temp = np.array(Weights[2], copy=True)
S = np.zeros(Weights[2].shape)
R = np.zeros(Weights[2].shape)
theta = np.zeros(Weights[2].shape)
f_prev = 0
it = 1
yhat, _, ip, fp = MLPclassificationPredict(Weights, X_old)
bias = np.ones((5000, 1))
yhat = yhat.T - np.max(yhat.T, axis=0)
s = np.exp(yhat)
yhat = (s / np.sum(s, axis=0)).T
f = fine_tune(yhat, yExpanded_old, Weights[2], penalty)
while (fine_tune(yhat, yExpanded_old, Weights[2], penalty) - f_prev) ** 2 >= 1e-5:
    f_prev = fine_tune(yhat, yExpanded_old, Weights[2], penalty)
    g = np.dot(np.concatenate((bias, fp[-1]), axis=1).T, yhat - yExpanded_old) + penalty * Weights[2]
    Weights[2], theta, S, R = adam(g, Weights[2], theta, S, R, it, 1e-5)
    yhat = np.dot(np.concatenate((bias, fp[-1]), axis=1), Weights[2])
    yhat = yhat.T - np.max(yhat.T, axis=0)
    s = np.exp(yhat)
    yhat = (s / np.sum(s, axis=0)).T
    it += 1

# Early Stopping
_, yhat, _, _ = MLPclassificationPredict(Weights, Xvalid)
if np.sum(yhat != (yvalid - 1)[:, 0]) / t - best < 0:
    Wbest = Weights[:]

# Evaluate test error after fine tuning
_, yhat, _, _ = MLPclassificationPredict(Wbest, Xtest)
te = np.sum(yhat != (ytest - 1)[:, 0]) / t2
print('Test error with final model = %f' % te)

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
