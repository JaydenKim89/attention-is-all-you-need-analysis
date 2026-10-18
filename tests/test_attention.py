"""Numerical tests for the NumPy attention reference.

  python -m unittest tests.test_attention -v

Gradient checks use float64, a tiny deterministic input, no dropout, and the
scalar loss L = sum(O * W) for a fixed random W (so dL/dO = W). Masked entries
are excluded structurally (probability exactly 0), so the loss is smooth in Q, K,
V at the test points.
"""

import unittest

import numpy as np
import torch

from src.attention_numpy import (
    attention_backward,
    attention_forward,
    causal_allowed,
    key_padding_allowed,
    masked_softmax,
    multihead_forward,
)

RTOL_FD = 1e-6   # central differences with eps = 1e-6 in float64
ATOL_FD = 1e-8
RTOL_AG = 1e-10  # same formula in float64, different code path (PyTorch autograd)
ATOL_AG = 1e-12


def rand(rng, *shape):
    return rng.standard_normal(shape)


def torch_attention(Q, K, V, allowed=None):
    """Independent reference using PyTorch tensor ops and autograd (float64)."""
    S = Q @ K.transpose(-1, -2) / Q.shape[-1] ** 0.5
    if allowed is not None:
        S = S.masked_fill(~allowed, float("-inf"))
    return torch.softmax(S, dim=-1) @ V


def finite_difference(f, X, eps=1e-6):
    """Central-difference gradient of scalar f with respect to every entry of X."""
    g = np.zeros_like(X)
    it = np.nditer(X, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = X[i]
        X[i] = old + eps
        fp = f()
        X[i] = old - eps
        fm = f()
        X[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        B, h, Lq, Lk, dk, dv = 2, 3, 4, 5, 6, 7
        self.Q, self.K, self.V = rand(self.rng, B, h, Lq, dk), rand(self.rng, B, h, Lk, dk), rand(self.rng, B, h, Lk, dv)

    def test_shapes_and_row_normalization(self):
        O, cache = attention_forward(self.Q, self.K, self.V)
        self.assertEqual(O.shape, (2, 3, 4, 7))
        self.assertEqual(cache["P"].shape, (2, 3, 4, 5))
        np.testing.assert_allclose(cache["P"].sum(-1), 1.0, rtol=0, atol=1e-12)
        self.assertTrue(np.all(cache["P"] >= 0))

    def test_matches_pytorch(self):
        O, _ = attention_forward(self.Q, self.K, self.V)
        ref = torch_attention(*(torch.tensor(a) for a in (self.Q, self.K, self.V))).numpy()
        np.testing.assert_allclose(O, ref, rtol=1e-12, atol=1e-12)

    def test_stable_softmax_large_finite_logits(self):
        s = np.array([[1000.0, 999.0, -1000.0], [-1e6, -1e6 + 1.0, -1e6 + 2.0]])
        p = masked_softmax(s)
        self.assertTrue(np.all(np.isfinite(p)))
        expected0 = np.exp([0.0, -1.0]) / np.exp([0.0, -1.0]).sum()
        np.testing.assert_allclose(p[0, :2], expected0, rtol=1e-12)
        self.assertEqual(p[0, 2], 0.0)  # exp(-2000) underflows to exactly 0
        np.testing.assert_allclose(p[1], masked_softmax(s[1:] + 1e6)[0], rtol=1e-12)
        # A naive exp(s)/sum(exp(s)) overflows here; the row-max shift avoids it.
        with np.errstate(over="ignore", invalid="ignore"):
            self.assertFalse(np.all(np.isfinite(np.exp(s[0]) / np.exp(s[0]).sum())))

    def test_key_padding_excludes_masked_keys(self):
        valid = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
        allowed = key_padding_allowed(valid)  # (B, 1, 1, Lk) broadcasts over heads and queries
        O, cache = attention_forward(self.Q, self.K, self.V, allowed)
        self.assertTrue(np.all(cache["P"][0, :, :, 3:] == 0.0))
        # Changing masked keys/values must not change the output.
        K2, V2 = self.K.copy(), self.V.copy()
        K2[0, :, 3:] += 100.0
        V2[0, :, 3:] -= 50.0
        O2, _ = attention_forward(self.Q, K2, V2, allowed)
        np.testing.assert_allclose(O, O2, rtol=0, atol=1e-12)

    def test_causal_mask_excludes_future_keys(self):
        Q, K, V = (rand(self.rng, 1, 1, 4, 3) for _ in range(3))
        allowed = causal_allowed(4)
        O, cache = attention_forward(Q, K, V, allowed)
        self.assertTrue(np.all(np.triu(cache["P"][0, 0], k=1) == 0.0))
        # Perturbing the last key/value changes only the last query's output.
        K2, V2 = K.copy(), V.copy()
        K2[..., 3, :] += 5.0
        V2[..., 3, :] += 5.0
        O2, _ = attention_forward(Q, K2, V2, allowed)
        np.testing.assert_allclose(O[..., :3, :], O2[..., :3, :], rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(O[..., 3, :], O2[..., 3, :]))

    def test_mask_broadcast_shape_is_checked(self):
        with self.assertRaises(ValueError):
            attention_forward(self.Q, self.K, self.V, np.ones((3, 3), dtype=bool))

    def test_all_masked_row_raises(self):
        allowed = np.ones((4, 5), dtype=bool)
        allowed[2] = False
        with self.assertRaises(ValueError):
            attention_forward(self.Q, self.K, self.V, allowed)

    def test_non_finite_input_raises(self):
        Q = self.Q.copy()
        Q[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            attention_forward(Q, self.K, self.V)

    def test_multihead_matches_per_head_loop(self):
        rng = np.random.default_rng(1)
        B, L, d, h = 2, 4, 8, 2
        X = rand(rng, B, L, d)
        Wq, Wk, Wv, Wo = (rand(rng, d, d) for _ in range(4))
        out, _ = multihead_forward(X, X, Wq, Wk, Wv, Wo, h)
        heads = []
        for i in range(h):
            cols = slice(i * d // h, (i + 1) * d // h)
            Oi, _ = attention_forward(X @ Wq[:, cols], X @ Wk[:, cols], X @ Wv[:, cols])
            heads.append(Oi)
        np.testing.assert_allclose(out, np.concatenate(heads, axis=-1) @ Wo, rtol=1e-12, atol=1e-12)


class GradientTests(unittest.TestCase):
    def check(self, allowed):
        rng = np.random.default_rng(42)
        Q, K, V, W = rand(rng, 3, 2), rand(rng, 4, 2), rand(rng, 4, 3), rand(rng, 3, 3)

        def loss():
            return float(np.sum(attention_forward(Q, K, V, allowed)[0] * W))

        _, cache = attention_forward(Q, K, V, allowed)
        manual = attention_backward(W, cache)

        tq, tk, tv = (torch.tensor(a, requires_grad=True) for a in (Q, K, V))
        ta = None if allowed is None else torch.tensor(allowed)
        (torch_attention(tq, tk, tv, ta) * torch.tensor(W)).sum().backward()
        autograd = (tq.grad.numpy(), tk.grad.numpy(), tv.grad.numpy())

        for name, X, g_manual, g_auto in zip("QKV", (Q, K, V), manual, autograd):
            g_fd = finite_difference(loss, X)
            np.testing.assert_allclose(g_manual, g_fd, rtol=RTOL_FD, atol=ATOL_FD, err_msg=f"d{name} vs FD")
            np.testing.assert_allclose(g_manual, g_auto, rtol=RTOL_AG, atol=ATOL_AG, err_msg=f"d{name} vs autograd")

    def test_unmasked_gradients(self):
        self.check(None)

    def test_masked_gradients(self):
        allowed = np.array([[1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=bool)
        self.check(allowed)

    def test_masked_entries_receive_no_gradient(self):
        rng = np.random.default_rng(3)
        Q, K, V = rand(rng, 3, 2), rand(rng, 4, 2), rand(rng, 4, 3)
        allowed = np.ones((3, 4), dtype=bool)
        allowed[:, 3] = False  # key 3 masked for every query
        _, cache = attention_forward(Q, K, V, allowed)
        _, dK, dV = attention_backward(np.ones((3, 3)), cache)
        self.assertTrue(np.all(dK[3] == 0.0) and np.all(dV[3] == 0.0))


if __name__ == "__main__":
    unittest.main()
