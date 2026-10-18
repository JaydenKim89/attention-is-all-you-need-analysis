# Attention Mathematics: Derivations and Checks

Paper references are to arXiv 1706.03762v7 (local PDF `references/Attention_Is_All_You_Need.pdf`). Each section names where it is implemented and which test could show it is wrong.

## 1. Shapes

| Symbol | Shape | Meaning |
|---|---|---|
| B, h | — | batch size, number of heads |
| L_q, L_k | — | query and key/value sequence lengths |
| d_model, d_k, d_v | — | model width, per-head key and value widths (paper: d_k = d_v = d_model / h) |
| Q | (B, h, L_q, d_k) | queries |
| K | (B, h, L_k, d_k) | keys |
| V | (B, h, L_k, d_v) | values |
| S = QKᵀ/√d_k | (B, h, L_q, L_k) | scaled compatibility scores |
| P = softmax(S) | (B, h, L_q, L_k) | each row sums to 1 over keys |
| O = PV | (B, h, L_q, d_v) | per-head outputs |

This project uses d_model = 64, h = 4, so d_k = d_v = 16 (`configs/baseline.json`).

## 2. Scaled dot-product attention (paper Eq. 1, Sec. 3.2.1)

O = softmax(QKᵀ / √d_k) V, with the softmax taken over the key axis.

**Why divide by √d_k (paper footnote 4, p.4).**
- Assume the components q_i, k_i of one query and one key are independent, with mean 0 and variance 1.
- Then q·k = Σᵢ qᵢkᵢ is a sum of d_k independent terms, each with mean 0 and variance E[qᵢ²]E[kᵢ²] = 1.
- So Var(q·k) = d_k, and dividing by √d_k gives variance 1.

These assumptions are idealized. After a learned projection of layer-normalized inputs, components are neither exactly unit-variance nor independent. In a trained model, queries and keys are deliberately aligned, so q·k is not mean-zero. The argument motivates the scale at initialization; it does not show that logits stay O(1) during training. This project does not ablate the scale (future work).

**Why large logits matter.** For one row p = softmax(s), the Jacobian is ∂pᵢ/∂sⱼ = pᵢ(δᵢⱼ − pⱼ), i.e. J = diag(p) − ppᵀ (Sec. 6). If one logit dominates, p approaches one-hot and every entry of J approaches 0. This is the "extremely small gradients" regime the paper refers to.

**Stable softmax.**
- For any per-row constant c, exp(sⱼ − c) / Σₖ exp(sₖ − c) = e⁻ᶜ exp(sⱼ) / (e⁻ᶜ Σₖ exp(sₖ)), so the probabilities are unchanged.
- With c = the row maximum over allowed entries, every exponent is ≤ 0 (no overflow) and at least one equals 0 (denominator ≥ 1, no division by zero).
- Implementation: `masked_softmax` in [src/attention_numpy.py](../src/attention_numpy.py). The model uses `torch.softmax`, which applies the same shift internally.
- Test: `test_stable_softmax_large_finite_logits` uses logits of ±1000 and −10⁶; it also confirms that the naive formula overflows on the same input.

## 3. Masks

A mask is a boolean `allowed` array broadcastable to (B, h, L_q, L_k); True means the query may attend to the key. Disallowed scores are set to −∞ before the softmax, so their probability is exactly 0 and they receive no gradient.

| Mask | Where | Shape | Rule |
|---|---|---|---|
| Source key padding | Encoder self-attention, decoder cross-attention | (B, 1, 1, L_s) | key j allowed iff source token j ≠ PAD |
| Target causal + padding | Decoder self-attention | (B, 1, L_t, L_t) | key j allowed iff j ≤ i and target input j ≠ PAD |

- Broadcasting is checked: the model calls `allowed.expand_as(scores)`, and NumPy uses `np.broadcast_to`. Both raise on incompatible shapes (`test_mask_broadcast_shape_is_checked`).
- **All-masked rows:** if a query row has no allowed key, the max is −∞ and softmax would produce NaN. Both implementations raise `ValueError` instead (`test_all_masked_row_raises`, NumPy and model).
- In this model such rows cannot occur for valid inputs. Source sequences have length ≥ 1, so every query sees a valid key, and decoder position 0 is BOS, which every target query may attend to. PAD *queries* still produce outputs, but those positions are excluded from the loss and from metrics.

## 4. Multi-head attention (Sec. 3.2.2)

1. X_q (B, L_q, d_model) · W^Q (d_model, h·d_k) → (B, L_q, h·d_k) → split → (B, h, L_q, d_k). The same applies to K and V from X_kv.
2. Attention per head → (B, h, L_q, d_v).
3. Concat heads → (B, L_q, h·d_v), then · W^O (h·d_v, d_model) → (B, L_q, d_model).

Column block i of W^Q is the paper's W^Q_i. Implementations: `multihead_forward` ([src/attention_numpy.py](../src/attention_numpy.py)) and `MultiHeadAttention` ([src/model.py](../src/model.py)), neither with projection biases.

Tests:
- `test_multihead_matches_per_head_loop`: block layout equals an explicit per-head loop.
- `test_attention_matches_numpy_reference`: the model equals the NumPy reference (float64, 1e-12).
- `test_attention_core_matches_pytorch_sdpa`: the attention core equals PyTorch's `scaled_dot_product_attention`, an independent reference with the same boolean-mask convention.

## 5. Worked example

Regenerate with `python -m src.worked_example`. Here d_k = d_v = 2 and L = 3:

Q = [[1,0],[0,1],[1,1]], K = [[1,0],[0,2],[1,1]], V = [[1,2],[3,0],[0,1]].

S = QKᵀ/√2 = [[0.7071, 0, 0.7071], [0, 1.4142, 0.7071], [0.7071, 1.4142, 1.4142]]

| | No mask | Causal mask |
|---|---|---|
| P | [[0.4011, 0.1978, 0.4011], [0.1400, 0.5760, 0.2840], [0.1978, 0.4011, 0.4011]] | [[1, 0, 0], [0.1956, 0.8044, 0], [0.1978, 0.4011, 0.4011]] |
| O = PV | [[0.9944, 1.2033], [1.8680, 0.5641], [1.4011, 0.7967]] | [[1, 2], [2.6089, 0.3911], [1.4011, 0.7967]] |
| ∂L/∂Q, L = ΣO | [[−0.1122, −0.1153], [−0.2313, 0.1751], [−0.2275, 0.1153]] | [[0, 0], [0, 0], [−0.2275, 0.1153]] |

The zeros in the causal ∂L/∂Q follow from Sec. 6:
- Row 0 has a single allowed key, so its softmax is constant (p = 1) and the Jacobian is 0.
- In row 1, ∂L/∂P = 1·Vᵀ = row sums of V = [3, 3, 1]. The two allowed keys have equal row sums, so dS = p ⊙ (dP − Σ p dP) = 0.

The last row is unaffected by the mask and matches the unmasked row.

## 6. Backward pass (manual VJP)

Take a scalar loss L and write G = ∂L/∂O. With c = 1/√d_k, apply the rules in reverse order of the forward pass:

| Forward step | Gradient |
|---|---|
| O = PV | ∂L/∂V = PᵀG, ∂L/∂P = GVᵀ |
| P = softmax(S), row-wise | with J = diag(p) − ppᵀ, ∂L/∂sⱼ = Σᵢ gᵢ pᵢ(δᵢⱼ − pⱼ) = pⱼ(gⱼ − Σᵢ gᵢpᵢ), i.e. dS = P ⊙ (dP − rowsum(dP ⊙ P)) |
| S = c·QKᵀ | ∂L/∂Q = c·dS·K, ∂L/∂K = c·dSᵀ·Q |

- A masked entry has P_ij = 0, so dS_ij = 0, and that key receives no gradient through query i (`test_masked_entries_receive_no_gradient`).
- Implementation: `attention_backward` in [src/attention_numpy.py](../src/attention_numpy.py).

**Verification** (`tests/test_attention.py::GradientTests`):
- Setup: float64, a fixed random input (seed 42), and the loss L = Σ O ⊙ W with W fixed, so G = W.
- **Finite differences:** central differences with ε = 10⁻⁶. Tolerance rtol 10⁻⁶, atol 10⁻⁸.
- **Autograd:** PyTorch float64. Tolerance rtol 10⁻¹⁰, atol 10⁻¹².
- Both checks run with and without a mask. No dropout is involved.
- The masked loss is smooth in Q, K, V because masking is structural; non-differentiable cases do not arise.

**Scope.** Only the attention core has a manual backward pass. The rest of the model relies on PyTorch autograd, checked indirectly by finite loss and gradients (`test_loss_ignores_pad_and_gradients_are_finite`) and by the overfit check.

## 7. Gradient flow through residual and normalization

Each sub-layer computes y = LN(x + Dropout(F(x))) (post-norm, Sec. 3.1). With dropout off:

∂y/∂x = J_LN(x + F(x)) · (I + J_F(x))

- The residual contributes the identity term I.
- In the post-norm arrangement, every layer's LayerNorm Jacobian multiplies the path, so the identity does not reach the input unmodified.
- This project keeps the paper's post-norm order and its warmup schedule. It does not test training without warmup or with pre-norm.

## 8. Positional information and a permutation symmetry

**Assumptions** (they hold for the `no_enc_pos` configuration in eval mode):
- A1. Encoder inputs are token embeddings only; no positional term (`enc_pos = false`).
- A2. Every encoder operation is either position-wise (embedding scale, FFN, residual add, LayerNorm, dropout as identity in eval) or self-attention with no position-dependent term. Its mask depends only on key validity (Sec. 3).
- A3. Padding is on the right, and a permutation π acts only on the valid source positions.

Let Π be the permutation matrix for π. Two facts:
- For a row-wise softmax, softmax(ΠAΠᵀ) = Π softmax(A) Πᵀ, since rows and columns are permuted consistently.
- softmax(aΠᵀ) = softmax(a)Πᵀ for a row vector a.

**Claim 1 (encoder equivariance).** Enc(πs) = Π Enc(s) on valid positions, and outputs at PAD positions are unchanged.
- Self-attention: Attn(ΠX) = softmax(ΠQKᵀΠᵀ)ΠV = Π softmax(QKᵀ)ΠᵀΠV = Π Attn(X), using ΠᵀΠ = I. The validity mask is unchanged under π, because π maps valid positions to valid positions.
- Position-wise maps commute with Π.
- Induction over layers gives the claim. PAD queries attend to the same set of valid keys, so their outputs are unchanged too.

**Claim 2 (cross-attention invariance).** For decoder queries q and memory M:
- CrossAttn(q, ΠM) = softmax(q(ΠK_M)ᵀ)ΠV_M = softmax(qK_Mᵀ)ΠᵀΠV_M = CrossAttn(q, M).
- Hence, given the same decoder input, every decoder logit is identical for s and πs.

**Consequence for this dataset.**
- Every example has a group partner with the same token multiset in a different order (`src/data.py`).
- Greedy decoding starts both partners from the same BOS, so the ablated model produces the same output sequence for both.
- Their targets differ, so at most one partner of each group can be exactly correct.
- This predicts `groups_both_correct = 0` for `no_enc_pos`, and that is what was measured on every split and seed ([04](04_training_and_results.md)).

**What the symmetry does not forbid.**
- Under teacher forcing the two partners have different decoder inputs (different target prefixes), so the model can use the prefix to rule out tokens it has already emitted.
- The best any order-blind predictor can do is p(next = t) = (remaining count of t) / (remaining length), the `multiset_oracle` in `src/evaluate.py`. It gives a teacher-forced loss of 0.963 nats/token on test.

**Numerical checks** (`tests/test_model.py::PermutationSymmetryTests`, float64, eval mode, atol 10⁻¹⁰):
- encoder outputs permute exactly;
- logits are invariant without encoder positions;
- logits differ by more than 10⁻³ with encoder positions;
- greedy outputs agree for a matched pair.

**Where the argument stops.**
- Encoder positional encoding, relative-position terms, or a position-dependent encoder mask would each break A1 or A2.
- In floating point, permuting the summation order changes logits at the rounding level, so an exact argmax tie could in principle break differently. None was observed.

## 9. Things this note does not establish

- That the scale 1/√d_k is optimal, or necessary at this model size (not ablated).
- That attention maps explain predictions causally (see [05](05_critical_analysis.md)).
- Anything about learning or generalization: the tests above check computation, not training outcomes.
