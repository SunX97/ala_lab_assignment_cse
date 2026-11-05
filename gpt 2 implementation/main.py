import numpy as np
from hyperparameters import L_MAX, L, H, D_E, D_MLP, D_ATTN, D_MID, N_V
from token_embedding import token_embedding
from layer_norm import layer_norm
from attention import attention, causal_mask
from mh_attention import mh_attention, init_mha_params

rng = np.random.default_rng(0)
length = 5

W_e = rng.normal(0, 0.02, (D_E, N_V))
W_p = rng.normal(0, 0.01, (D_E, L_MAX))
tokens = rng.integers(0, N_V, length)

X = np.stack([token_embedding(t, W_e) + W_p[:, i] for i, t in enumerate(tokens)], axis=1)
print("embedded:", X.shape)

gamma = np.ones(D_E)
beta = np.zeros(D_E)
X_ln = layer_norm(X, gamma, beta)
print("layer_norm:", X_ln.shape, X_ln[:, 0].mean().round(6), X_ln[:, 0].std().round(4))

mask = causal_mask(length)
heads, Wo, bo = init_mha_params()
out_single = attention(X_ln, X_ln, heads[0], mask)
print("attention:", out_single.shape)

out = mh_attention(X_ln, X_ln, (heads, Wo, bo), mask)
print("mh_attention:", out.shape)

X2 = X_ln.copy()
X2[:, -1] = rng.normal(size=D_E)
out2 = mh_attention(X2, X2, (heads, Wo, bo), mask)
print("causal check:", np.allclose(out[:, :-1], out2[:, :-1]))
