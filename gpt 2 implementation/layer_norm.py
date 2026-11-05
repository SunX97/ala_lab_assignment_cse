import numpy as np

def layer_norm(e, gamma, beta, eps=1e-5):
    m = e.mean(axis=0, keepdims=True)
    v = ((e - m) ** 2).mean(axis=0, keepdims=True)
    e_hat = (e - m) / np.sqrt(v + eps)
    if e.ndim == 1:
        return e_hat * gamma + beta
    return e_hat * gamma[:, None] + beta[:, None]
