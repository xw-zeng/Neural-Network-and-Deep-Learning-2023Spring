import numpy as np


def adam(g_t, Weights_t, theta_t, S_t, R_t, t, alpha):
    # Internal parameters
    beta1 = 0.9
    beta2 = 0.999
    rho = 0.95
    # Update by Adam
    S_t = beta1 * S_t + (1 - beta1) * g_t
    R_prev = np.array(R_t, copy = True)
    R_t = (beta2 * R_t + (1 - beta2) * (g_t ** 2))
    if (R_t ** 2).sum() == np.nan or (R_t ** 2).sum() < (R_prev ** 2).sum():
        R_t = R_prev
    theta_t = rho * theta_t - alpha / (1 - beta1 ** t) * S_t / (np.sqrt(R_t / (1 - beta2 ** t) + 1e-7))
    Weights_t = Weights_t + theta_t
    return Weights_t, theta_t, S_t, R_t
