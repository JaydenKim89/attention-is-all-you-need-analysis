"""Functional tests for the PyTorch model, masks, target shift, and data splits.

  python -m unittest tests.test_model -v

Numerical comparisons run the model in float64 with dropout disabled (eval mode),
so invariance checks use tight tolerances.
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.attention_numpy import multihead_forward
from src.data import DEFAULTS, generate, make_batch, validate
from src.model import BOS, EOS, PAD, ModelConfig, MultiHeadAttention, Transformer, token_cross_entropy

ATOL = 1e-10


def tiny_model(enc_pos=True, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=11, d_model=16, n_heads=2, d_ff=32, n_enc_layers=2, n_dec_layers=2,
                      dropout=0.1, max_len=32, enc_pos=enc_pos)
    return Transformer(cfg).double().eval()


class ShiftAndMaskTests(unittest.TestCase):
    def test_target_shift_and_eos(self):
        src, tgt_in, tgt_out = make_batch([{"src": [3, 4, 5], "tgt": [5, 4, 3]},
                                           {"src": [6, 7], "tgt": [7, 6]}])
        self.assertEqual(src.tolist(), [[3, 4, 5], [6, 7, PAD]])
        self.assertEqual(tgt_in.tolist(), [[BOS, 5, 4, 3], [BOS, 7, 6, PAD]])
        self.assertEqual(tgt_out.tolist(), [[5, 4, 3, EOS], [7, 6, EOS, PAD]])

    def test_future_target_tokens_do_not_affect_earlier_predictions(self):
        m = tiny_model()
        src, tgt_in, _ = make_batch([{"src": [3, 4, 5, 6], "tgt": [6, 5, 4, 3]}])
        base = m(src, tgt_in)
        for t in range(1, tgt_in.shape[1]):
            changed = tgt_in.clone()
            changed[0, t] = 9  # replace the input token at position t
            out = m(src, changed)
            torch.testing.assert_close(out[:, :t], base[:, :t], rtol=0, atol=ATOL)
            self.assertFalse(torch.allclose(out[:, t:], base[:, t:]))

    def test_source_right_padding_does_not_change_valid_outputs(self):
        m = tiny_model()
        src, tgt_in, _ = make_batch([{"src": [3, 4, 5], "tgt": [5, 4, 3]}])
        padded = torch.cat([src, torch.full((1, 4), PAD)], dim=1)  # same tokens, same positions
        torch.testing.assert_close(m(padded, tgt_in), m(src, tgt_in), rtol=0, atol=ATOL)

    def test_loss_ignores_pad_and_gradients_are_finite(self):
        m = tiny_model()
        m.train()
        src, tgt_in, tgt_out = make_batch([{"src": [3, 4, 5, 6], "tgt": [6, 5, 4, 3]},
                                           {"src": [7, 8], "tgt": [8, 7]}])
        logits = m(src, tgt_in)
        loss = token_cross_entropy(logits, tgt_out)
        valid = tgt_out != PAD
        manual = F.cross_entropy(logits[valid], tgt_out[valid])  # mean over 5 + 3 = 8 tokens
        torch.testing.assert_close(loss, manual)
        self.assertEqual(int(valid.sum()), 8)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_all_masked_row_raises(self):
        attn = MultiHeadAttention(8, 2).double()
        x = torch.randn(1, 3, 8, dtype=torch.float64)
        allowed = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            attn(x, x, allowed)

    def test_greedy_first_step_matches_teacher_forced_argmax(self):
        m = tiny_model()
        src = torch.tensor([[3, 4, 5]])
        first = m(src, torch.tensor([[BOS]]))[0, -1].argmax().item()
        outs, _ = m.greedy_decode(src, max_new_tokens=1)
        self.assertEqual(outs[0][:1] if outs[0] else [EOS], [first])


class ReferenceTests(unittest.TestCase):
    def test_attention_matches_numpy_reference(self):
        torch.manual_seed(0)
        attn = MultiHeadAttention(8, 2).double()
        x = torch.randn(2, 4, 8, dtype=torch.float64)
        allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
        ours = attn(x, x, allowed).detach().numpy()
        # nn.Linear computes x @ W^T, so the NumPy weights are the transposes.
        W = [l.weight.detach().numpy().T for l in (attn.w_q, attn.w_k, attn.w_v, attn.w_o)]
        ref, _ = multihead_forward(x.numpy(), x.numpy(), *W, h=2, allowed=allowed.numpy())
        np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)

    def test_attention_core_matches_pytorch_sdpa(self):
        # Independent reference for the core only; boolean attn_mask True = may attend.
        torch.manual_seed(1)
        q, k, v = (torch.randn(2, 2, 4, 3, dtype=torch.float64) for _ in range(3))
        allowed = torch.ones(4, 4, dtype=torch.bool).tril()
        ours = torch.softmax((q @ k.transpose(-1, -2) / 3 ** 0.5).masked_fill(~allowed, float("-inf")), -1) @ v
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        torch.testing.assert_close(ours, ref, rtol=1e-12, atol=1e-12)


class PermutationSymmetryTests(unittest.TestCase):
    """Without encoder positional encoding, permuting valid source positions permutes
    encoder outputs and leaves cross-attention, hence all decoder logits, unchanged."""

    def setUp(self):
        self.src = torch.tensor([[3, 4, 5, 6, 7, PAD, PAD]])
        self.perm = torch.tensor([3, 0, 4, 1, 2, 5, 6])  # permutes only the 5 valid positions
        self.tgt_in = torch.tensor([[BOS, 7, 6, 5]])

    def test_encoder_outputs_permute_and_logits_are_invariant_without_enc_pos(self):
        m = tiny_model(enc_pos=False)
        src_p = self.src[:, self.perm]
        mem, _ = m.encode(self.src)
        mem_p, _ = m.encode(src_p)
        torch.testing.assert_close(mem_p[:, :5], mem[:, self.perm[:5]], rtol=0, atol=ATOL)
        torch.testing.assert_close(m(src_p, self.tgt_in), m(self.src, self.tgt_in), rtol=0, atol=ATOL)

    def test_symmetry_is_broken_with_enc_pos(self):
        m = tiny_model(enc_pos=True)
        out, out_p = m(self.src, self.tgt_in), m(self.src[:, self.perm], self.tgt_in)
        self.assertGreater((out - out_p).abs().max().item(), 1e-3)

    def test_greedy_outputs_identical_for_a_matched_group_without_enc_pos(self):
        m = tiny_model(enc_pos=False)
        a, b = torch.tensor([[3, 3, 5, 8, 4]]), torch.tensor([[8, 3, 4, 3, 5]])  # same multiset
        self.assertEqual(m.greedy_decode(a, 8)[0], m.greedy_decode(b, 8)[0])


class DataTests(unittest.TestCase):
    def test_default_dataset_is_valid_and_deterministic(self):
        a, b = generate(DEFAULTS), generate(DEFAULTS)
        self.assertEqual(a, b)
        counts = validate(a)
        self.assertEqual(counts, {"train": 8000, "dev": 1000, "test": 1000, "test_long": 500})

    def test_validate_rejects_group_split_across_splits(self):
        splits = generate(dict(DEFAULTS, train_groups=3, dev_groups=2, test_groups=2, long_test_groups=1))
        moved = splits["train"].pop()
        splits["dev"].append(moved)  # its group partner stays in train
        with self.assertRaises(AssertionError):
            validate(splits)


if __name__ == "__main__":
    unittest.main()
