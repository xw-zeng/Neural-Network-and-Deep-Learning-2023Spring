import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from DataAugmentation import DataAugmentation

m = loadmat('../data/digits.mat')
X = m['X']
y = m['y']
_, axes = plt.subplots(3, 9, figsize=(10, 3.5))
axes = axes.flatten()
idx = [36, 42, 0, 1, 44, 17, 2, 5, 3]
imgs0 = X[idx, :]
imgs1 = np.array([DataAugmentation(img, False) for img in imgs0])
imgs2 = np.array([DataAugmentation(img, True) for img in imgs0])
imgs = np.r_[imgs0, imgs1, imgs2]
for i, (ax, img) in enumerate(zip(axes, imgs)):
    ax.imshow(img.reshape((16, 16), order='F'), cmap=plt.cm.binary)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i < 9:
        ax.set_title(y[idx[i], 0])
plt.show()
