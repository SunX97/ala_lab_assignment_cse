import numpy as np

def causal_mask(length):
    return np.triu(np.ones((length, length), dtype=bool))

def softmax(S):
    S = S - S.max(axis=0, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=0, keepdims=True)

def attention(X, Z, params, mask):
    Wq, bq, Wk, bk, Wv, bv = params
    d_attn = Wq.shape[0]
    Q = Wq @ X + bq[:, None]
    K = Wk @ Z + bk[:, None]
    V = Wv @ Z + bv[:, None]
    S = K.T @ Q / np.sqrt(d_attn)
    S = np.where(mask, S, -np.inf)
    return V @ softmax(S)
