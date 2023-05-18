import numpy as np


def standardizeCols(M, mu=None, sigma2=None):
    """Standardize columns."""
    # Make each column of M be zero mean, std 1.
    # If mu, sigma2 are omitted, they are computed from M.
    if mu is None:
        mu = M.mean(axis = 0)
    if sigma2 is None:
        sigma2 = np.std(M,axis=0)
    S = (M - mu) / sigma2
    return S, mu, sigma2
