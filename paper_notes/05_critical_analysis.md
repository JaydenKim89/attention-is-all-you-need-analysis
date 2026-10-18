# Critical Analysis

## 1. Cost of self-attention, stated precisely

The earlier version of this note said self-attention "requires O(n²d) computation per layer and O(n²) memory to store the n × n attention matrix". That is the attention core only. A full layer has several distinct costs.

For one self-attention sub-layer with sequence length n, width d = d_model, h heads, d_k = d_v = d/h, counted in multiply-adds per sequence:

| Part | Multiply-adds | Parameters | Depends on n as |
|---|---|---|---|
| Q, K, V, output projections | 4·n·d² | 4·d² (no bias here) | linear |
| Scores QKᵀ, all heads | h·n²·d_k = n²·d | 0 | quadratic |
| Weighted sum PV, all heads | n²·d | 0 | quadratic |
| Softmax | O(h·n²) exponentials and divisions | 0 | quadratic |
| FFN (separate sub-layer) | 2·n·d·d_ff | 2·d·d_ff + d_ff + d | linear |

- **The paper's figure.** Table 1 (p.6) lists O(n²·d) for self-attention: the core term. The projections, O(n·d²), and the FFN are separate. The paper notes self-attention is cheaper than recurrence when n < d (Sec. 4).
- **Memory.** This implementation materializes the scores and probabilities explicitly, as a (B, h, L_q, L_k) tensor per attention call, and autograd keeps them for the backward pass. Activation storage for attention therefore grows as B·h·n² per layer. Parameter count does not depend on n.
- **At this project's sizes** (n ≤ 17 target tokens, d = 64, d_ff = 256):
  - projections: 4·17·64² ≈ 2.8·10⁵ multiply-adds;
  - attention core: 2·17²·64 ≈ 3.7·10⁴;
  - FFN: 2·17·64·256 ≈ 5.6·10⁵.

  The quadratic term is the smallest of the three here. These are operation counts from the formulas, not measured runtimes. No kernel-level benchmark was run. The recorded wall-clock per training run (`train_seconds` in each `results/runs/*/summary.json`) includes Python overhead, data batching, and evaluation.

## 2. Reading the results by kind of evidence

| Statement | Kind | Where |
|---|---|---|
| Without encoder positions, permuting valid source positions permutes encoder outputs and leaves decoder logits unchanged | Mathematical property under assumptions A1–A3 | [03](03_attention_math.md#8-positional-information-and-a-permutation-symmetry) |
| The implementation satisfies that property to 10⁻¹⁰ in float64 | Numerical check on specific inputs | `tests/test_model.py::PermutationSymmetryTests` |
| Masks, target shift, PAD handling, and gradients behave as specified | Functional tests | `tests/` (25 tests) |
| Baseline reaches near-perfect exact match on in-distribution test; the ablation does not; neither handles lengths 13–16 | Measured learning results, 3 seeds, fixed budget | [04](04_training_and_results.md#5-results-this-repository) |
| The ablation's loss sits close to the best possible order-blind loss | Measured result compared with a computed reference | [04](04_training_and_results.md#5-results-this-repository) |
| The baseline fails on longer inputs because positions beyond 12–13 were never seen in training | Interpretation, untested | below |

## 3. Optimization vs. capacity vs. information

A failure to learn can come from optimization (the right function is representable but not found), capacity (it is not representable at this size), or information (the input does not determine the answer). The checks used to separate these:

- **Implementation errors first.** Tests cover the target shift, causal isolation, padding invariance, loss reduction, finite gradients, and agreement with independent references.
- **Optimization.** The baseline drives teacher-forced loss on 64 training examples to 0.006 within 600 steps (overfit check), and reaches near-zero dev loss on the full task. The training procedure can fit this model class.
- **Information.**
  - The ablated model's teacher-forced loss plateaus near 1.0 nat/token, close to the order-blind oracle (about 0.95–0.96 on dev and test).
  - Its greedy outputs have the correct multiset about 95% of the time, but the correct order about 1% of the time.
  - Matched pairs never both succeed.
  
  This is what the symmetry predicts: content is available, order is not. It is consistent with an information limit. It does not prove that no amount of extra training could change anything; the symmetry argument is what rules order out.

## 4. Fitting vs. generalization

- **Within lengths 4–12**, baseline train, dev, and test results agree closely. Train, dev, and test contain disjoint permutation groups, and no source sequence appears in two splits (`src/data.py::validate`).
- **On lengths 13–16** (`test_long`, a separate distribution shift), the baseline gets no sequence right, and its teacher-forced loss rises to about 7 nats/token.
- **The paper's hope does not hold here.** The paper chose sinusoids partly because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training" (Sec. 3.5). On this task, model size, and budget, that did not happen.
- **One untested explanation:** the model learned to use the specific absolute positions it saw, and positions 13–17 behave differently. This was not probed: no length-curriculum, relative-position, or learned-position comparison was run.
- **The ablation degrades more gracefully** in multiset accuracy on longer inputs (about 40% vs. about 0%). That is consistent with it relying less on absolute source positions. Its exact match is still 0.

## 5. Attention maps: what can be read from them

- **What is plotted.** `results/figures/attention.png` shows the probabilities of the last decoder layer's cross-attention, for the first test example, under teacher forcing, one panel per head, from `baseline_s0`.
- **What it shows.** At decoder step t, the heads concentrate on source position L−1−t: the token to be emitted next. That pattern is consistent with reversal.
- **What it cannot show:**
  - that the model computes its prediction *because of* that pattern (no intervention was done);
  - what earlier layers or other examples do;
  - anything about linguistic structure.

  One example and one layer is an illustration, not an explanation.

## 6. Limits of this project

- **Toy task.** A synthetic reversal task with an 8-symbol vocabulary. It is not machine translation, and no BLEU, training-cost, or original-scale result is reproduced.
- **Small scale.** One small configuration (232,640 parameters), one budget (3,000 steps), and three seeds per arm. The seeds vary initialization only, and the data split is fixed.
- **One ablation.** Removing encoder positional encoding. Head count, the 1/√d_k scale, label smoothing, normalization order, and learned positions are not varied.
- **Greedy decoding only.** The paper used beam search; it is not implemented.
- **Project deviations from the paper** are listed in [04](04_training_and_results.md#3-configuration-used).
