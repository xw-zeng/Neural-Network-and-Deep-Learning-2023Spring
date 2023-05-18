import numpy as np


def fine_tune(yhat, y, w, penalty):
    return np.multiply(- np.log(yhat + 1e-6), y).sum() + penalty / 2 * (np.linalg.norm(w, 2) ** 2)
