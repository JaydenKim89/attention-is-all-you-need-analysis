"""NumPy reference for scaled dot-product attention: forward and manual backward.

Shapes (any leading batch/head dimensions "..." are allowed):
  Q: (..., Lq, dk)   K: (..., Lk, dk)   V: (..., Lk, dv)
  allowed: boolean, broadcastable to (..., Lq, Lk); True = query may attend to key
  S = Q K^T / sqrt(dk): (..., Lq, Lk)   P = softmax(S, masked): (..., Lq, Lk)
  O = P V: (..., Lq, dv)

Invalid-input policy: non-finite Q/K/V and query rows with no allowed key raise
ValueError instead of returning NaNs. Derivations: paper_notes/03_attention_math.md.
"""

import numpy as np


def _check_finite(**arrays):
    for name, a in arrays.items():
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite values")


def masked_softmax(scores, allowed=None):
    """Row softmax over the last axis, with disallowed entries given probability 0.

    Subtracting the row maximum (over allowed entries) leaves the probabilities
    unchanged because softmax(s - c) = softmax(s) for any constant c per row,
    and keeps every exponent <= 0, so exp() cannot overflow.
    """
    if allowed is None:
        allowed = np.ones(scores.shape, dtype=bool)
    allowed = np.broadcast_to(np.asarray(allowed, dtype=bool), scores.shape)
    if np.any(~allowed.any(axis=-1)):
        raise ValueError("a query row has no allowed key (all-masked row)")
    masked = np.where(allowed, scores, -np.inf)
    row_max = masked.max(axis=-1, keepdims=True)  # finite: each row has an allowed entry
    e = np.where(allowed, np.exp(masked - row_max), 0.0)
    return e / e.sum(axis=-1, keepdims=True)


def attention_forward(Q, K, V, allowed=None):
    """Return (O, cache) for O = softmax(Q K^T / sqrt(dk), masked) V."""
    _check_finite(Q=Q, K=K, V=V)
    dk = Q.shape[-1]
    if K.shape[-1] != dk or K.shape[-2] != V.shape[-2]:
        raise ValueError(f"incompatible shapes Q{Q.shape} K{K.shape} V{V.shape}")
    scale = 1.0 / np.sqrt(dk)
    S = (Q @ np.swapaxes(K, -1, -2)) * scale
    P = masked_softmax(S, allowed)
    O = P @ V
    return O, {"Q": Q, "K": K, "V": V, "P": P, "scale": scale}


def attention_backward(dO, cache):
    """Manual vector-Jacobian products for attention_forward.

    Given dO = dL/dO:
      dV = P^T dO
      dP = dO V^T
      dS = P * (dP - sum_j(dP * P))          (softmax VJP, row-wise)
      dQ = scale * dS K,   dK = scale * dS^T Q
    Disallowed entries have P = 0, so their dS is 0 and no gradient flows there.
    """
    Q, K, V, P, scale = cache["Q"], cache["K"], cache["V"], cache["P"], cache["scale"]
    dV = np.swapaxes(P, -1, -2) @ dO
    dP = dO @ np.swapaxes(V, -1, -2)
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
    dQ = scale * (dS @ K)
    dK = scale * (np.swapaxes(dS, -1, -2) @ Q)
    return dQ, dK, dV


def causal_allowed(Lq, Lk=None):
    """allowed[i, j] = (j <= i): position i may attend to itself and earlier keys."""
    Lk = Lq if Lk is None else Lk
    return np.tril(np.ones((Lq, Lk), dtype=bool))


def key_padding_allowed(key_valid):
    """(B, Lk) validity -> (B, 1, 1, Lk) mask that broadcasts over heads and queries."""
    return np.asarray(key_valid, dtype=bool)[:, None, None, :]


def split_heads(X, h):
    """(B, L, h*d) -> (B, h, L, d)."""
    B, L, D = X.shape
    return X.reshape(B, L, h, D // h).transpose(0, 2, 1, 3)


def merge_heads(X):
    """(B, h, L, d) -> (B, L, h*d), the paper's Concat(head_1, ..., head_h)."""
    B, h, L, d = X.shape
    return X.transpose(0, 2, 1, 3).reshape(B, L, h * d)


def multihead_forward(Xq, Xkv, Wq, Wk, Wv, Wo, h, allowed=None):
    """Multi-head attention without biases, as in Eq. (MultiHead) of the paper.

    Xq: (B, Lq, d_model), Xkv: (B, Lk, d_model)
    Wq, Wk: (d_model, h*dk), Wv: (d_model, h*dv), Wo: (h*dv, d_model)
    Column block i of Wq/Wk/Wv is the paper's per-head projection W_i.
    """
    Q = split_heads(Xq @ Wq, h)
    K = split_heads(Xkv @ Wk, h)
    V = split_heads(Xkv @ Wv, h)
    O, cache = attention_forward(Q, K, V, allowed)
    return merge_heads(O) @ Wo, cache
