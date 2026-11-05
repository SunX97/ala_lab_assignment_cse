import numpy as np
from attention import attention
from hyperparameters import H, D_E, D_ATTN, D_MID

def init_mha_params(seed=0):
    rng = np.random.default_rng(seed)
    heads = []
    for _ in range(H):
        Wq = rng.normal(0, 0.02, (D_ATTN, D_E))
        bq = np.zeros(D_ATTN)
        Wk = rng.normal(0, 0.02, (D_ATTN, D_E))
        bk = np.zeros(D_ATTN)
        Wv = rng.normal(0, 0.02, (D_MID, D_E))
        bv = np.zeros(D_MID)
        heads.append((Wq, bq, Wk, bk, Wv, bv))
    Wo = rng.normal(0, 0.02, (D_E, H * D_MID))
    bo = np.zeros(D_E)
    return heads, Wo, bo

def mh_attention(X, Z, params, mask):
    heads, Wo, bo = params
    Y = np.concatenate([attention(X, Z, p, mask) for p in heads], axis=0)
    return Wo @ Y + bo[:, None]
