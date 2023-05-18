import numpy as np
from scipy.io import loadmat, savemat
from DataAugmentation import DataAugmentation

m = loadmat('../data/digits.mat')
X = m['X']
y = m['y']
Xvalid = m['Xvalid']
Xtest = m['Xtest']
yvalid = m['yvalid']
ytest = m['ytest']
X0 = np.r_[X[:, :], X[:, :], X[:, :], X[:, :], X[:, :]]
X1 = np.array([DataAugmentation(img, True) for img in X0])
X2 = np.array([DataAugmentation(img, True) for img in X0])
X3 = np.array([DataAugmentation(img, True) for img in X0])
X4 = np.array([DataAugmentation(img, False) for img in X0[:20000, :]])
X_new = np.r_[X, X1, X2, X3, X4]
y0 = np.r_[y[:, :], y[:, :], y[:, :], y[:, :], y[:, :]]
y_new = np.r_[y, y0, y0, y0]
savemat('../data/mynewdigits.mat', {'X': X_new, 'y': y_new,
                                    'Xvalid': Xvalid, 'yvalid': yvalid,
                                    'Xtest': Xtest, 'ytest': ytest})
