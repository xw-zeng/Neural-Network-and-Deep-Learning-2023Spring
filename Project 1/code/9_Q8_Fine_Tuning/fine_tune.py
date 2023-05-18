import numpy as np


def fine_tune(yhat, y):
    return np.multiply(- np.log(yhat + 1e-6), y).sum()
