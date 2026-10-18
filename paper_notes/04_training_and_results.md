# Training and Results

Two different things are reported here, and they must not be confused:

1. what the paper's authors report for machine translation (Sec. 1), and
2. what this repository measured on a small synthetic reversal task (Secs. 4–5).

Nothing in Sec. 5 reproduces the paper's BLEU scores, training cost, or scale.

## 1. The paper's experiments (arXiv 1706.03762v7)

| Item | Paper | Location |
|---|---|---|
| Data | WMT 2014 EN-DE, about 4.5M sentence pairs, BPE about 37K shared tokens; EN-FR, 36M sentences, 32K word-pieces | Sec. 5.1, p.7 |
| Batching | About 25,000 source + 25,000 target tokens per batch | Sec. 5.1 |
| Hardware and steps | 8 P100 GPUs. Base: 100K steps, about 0.4 s/step, 12 h. Big: 300K steps, 3.5 days | Sec. 5.2 |
| Optimizer | Adam, β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹; lrate = d_model⁻⁰·⁵ · min(step⁻⁰·⁵, step · warmup⁻¹·⁵), warmup = 4000 | Sec. 5.3, Eq. 3 |
| Regularization | Residual dropout 0.1 on sub-layer outputs and on embedding + position sums; label smoothing 0.1 | Sec. 5.4, p.8 |
| Decoding | Beam size 4, length penalty 0.6; checkpoint averaging (last 5 base, last 20 big) | Sec. 6.1 |
| Translation results | Base: 27.3 BLEU EN-DE, 38.1 EN-FR. Big: 28.4 EN-DE, 41.8 EN-FR (Table 2). The text of Sec. 6.1 gives 41.0 for big EN-FR, so the paper is internally inconsistent on that number. | Table 2, p.8; Sec. 6.1 |
| Variations | Single head is 0.9 BLEU worse than the best setting; smaller d_k hurts; bigger models and dropout help; learned positional embeddings give nearly identical results to sinusoids | Table 3, p.9 |
| Parsing | WSJ only, 91.3 F1; semi-supervised, 92.7 F1 | Table 4, p.10 |

## 2. Learning setup in this repository

| Concept | Precise statement | Assumptions | Implemented in | What would show it wrong |
|---|---|---|---|---|
| Autoregressive factorization | p(y \| x) = ∏ₜ₌₁ᵀ⁺¹ p(yₜ \| y₍<ₜ₎, x), with y_{T+1} = EOS | Exact by the chain rule; the model must make prediction t depend only on y₍<ₜ₎ | `Transformer.tgt_mask` (causal) | `test_future_target_tokens_do_not_affect_earlier_predictions` |
| Maximum likelihood and cross-entropy | Maximizing Σ log p(y \| x) equals minimizing Σₜ −log p(yₜ \| ·). The loss is the mean over all non-PAD target tokens in a batch, EOS included, so longer sequences carry more weight. | Training examples treated as independent draws | `token_cross_entropy` | `test_loss_ignores_pad_and_gradients_are_finite` (equals a manual mean over 8 valid tokens) |
| Teacher forcing and shift | tgt_in = [BOS, y₁…y_T], tgt_out = [y₁…y_T, EOS]; position t is trained to predict tgt_out[t] from tgt_in[≤ t] | — | `src/data.py::make_batch` | `test_target_shift_and_eos` |
| PAD handling | PAD is excluded from the loss, from token accuracy, and from attention keys | Right padding | `token_cross_entropy`, `src_mask`, `tgt_mask` | `test_source_right_padding_does_not_change_valid_outputs` |
| Training vs. inference | Training conditions on the true prefix. Greedy decoding conditions on the model's own outputs, so one error changes every later input. | — | `Transformer.greedy_decode` | `test_greedy_first_step_matches_teacher_forced_argmax`; teacher-forced vs. exact accuracy in Sec. 5 |
| Optimization | Adam with the paper's β, ε and the Eq. 3 schedule (Sec. 3) | Empirical; no convergence guarantee | `src/train.py::noam_lr` | Overfit check (Sec. 4); non-finite loss stops training |
| Generalization and leakage | Train, dev, and test contain disjoint permutation groups and no repeated source. Dev chooses the configuration; test is used once after freezing; `test_long` is a separate shift. | Fixed seed-1234 split | `src/data.py::validate` | `DataTests`; `validate` raises on any leak |
| Token vs. sequence success | Exact match needs every token and the EOS; token accuracy can be high while exact match is near 0 | — | `src/evaluate.py` | Sec. 5, ablation row |
| Inductive bias | Without positional encoding, encoder self-attention is permutation-equivariant and cross-attention is permutation-invariant | A1–A3 in [03 §8](03_attention_math.md#8-positional-information-and-a-permutation-symmetry) | `enc_pos` switch | `PermutationSymmetryTests`; `groups_both_correct` |

## 3. Configuration used

From `configs/baseline.json`. The ablation `configs/no_enc_pos.json` differs only in `enc_pos: false`.

| Setting | This repository | Paper base model |
|---|---|---|
| Layers (enc / dec) | 2 / 2 | 6 / 6 |
| d_model, h, d_k = d_v, d_ff | 64, 4, 16, 256 | 512, 8, 64, 2048 |
| Parameters | 232,640 | about 65M |
| Normalization, activation | Post-norm, ReLU | Same |
| Embeddings | Shared across encoder, decoder, and output projection; × √d_model | Same (Sec. 3.4) |
| Positional encoding | Sinusoidal; baseline in both stacks, ablation decoder only | Sinusoidal, both stacks |
| Dropout | 0.1 on sub-layer outputs and embedding sums; none on attention probabilities | 0.1 (Sec. 5.4) |
| Label smoothing | **None** (project choice: keeps the loss a plain NLL, comparable to the oracle in Sec. 5) | 0.1 |
| Optimizer | Adam (0.9, 0.98, 10⁻⁹), Eq. 3, **warmup 400** (scaled down; project choice) | Warmup 4000 |
| Batch | 128 sequences | About 25K tokens |
| Budget | 3,000 steps | 100K steps |
| Precision | float32, TF32 disabled | Not stated |
| Initialization | Embeddings N(0, d_model⁻⁰·⁵); Xavier-uniform linear layers (project choice) | Not stated |
| Decoding | Greedy, stop at EOS or after len(src) + 5 tokens | Beam 4 |

## 4. Procedure

1. **Dataset.** `python -m src.data --seed 1234` writes `data/reversal_v1/`; checksums are in `results/dataset_manifest.json`.
   - 8 symbols with repeats allowed; lengths 4–12.
   - Every source has one partner with the same multiset and length, and partners share a split.
   - 8,000 train / 1,000 dev / 1,000 test examples, plus 500 `test_long` examples (lengths 13–16).
   - Regenerating with the same seed gave identical checksums.
2. **Overfit check.** Baseline config on the first 64 training examples for 600 steps (`overfit_check_64`): final loss 0.0064, token accuracy 1.0, and greedy exact match 1.0, all on those same 64 examples.
3. **Pilot.** Seed 0 of each config was trained once and evaluated on dev only. The configs were not changed after the pilot, so the pilot runs are identical to the recorded seed-0 runs.
4. **Main runs.** Seeds 0, 1, 2 per config; seeds change initialization and batch order only. Seed 0 reproduced the pilot's metrics exactly on this machine.
5. **Test evaluation.** `test` and `test_long` were evaluated once, after the configs were frozen. Two descriptive metrics (length match and multiset match) were added to the evaluator afterwards, and the same frozen checkpoints were re-evaluated. Nothing was retrained or re-selected.

Every `results/runs/*/summary.json` records the git commit, a `git_dirty` flag, a SHA-256 of `src/*.py` and `configs/*.json` at run time, and the environment.

## 5. Results (this repository)

From `results/summary.csv`: mean over 3 seeds, with min and max where shown. Loss is in nats per non-PAD target token; exact match is greedy.

| Config | Split | Loss | Token acc | Exact acc [min, max] | Correct multiset | Groups with both members correct (max over seeds) |
|---|---|---|---|---|---|---|
| baseline | dev | 0.0018 | 0.9994 | 0.995 [0.987, 1.000] | 0.995 | 500 / 500 |
| baseline | test | 0.0010 | 0.9998 | 0.998 [0.996, 1.000] | 0.998 | 500 / 500 |
| baseline | test_long | 7.15 | 0.246 | 0.000 [0, 0] | 0.001 | 0 / 250 |
| no_enc_pos | dev | 0.999 | 0.531 | 0.010 [0.009, 0.012] | 0.957 | 0 / 500 |
| no_enc_pos | test | 1.011 | 0.524 | 0.009 [0.005, 0.011] | 0.954 | 0 / 500 |
| no_enc_pos | test_long | 1.485 | 0.408 | 0.000 [0, 0] | 0.399 | 0 / 250 |

Reference predictors, using the same PAD/EOS conventions (token accuracy over target tokens plus EOS):

| Predictor | dev | test | test_long |
|---|---|---|---|
| Copy source (exact = palindromes only) | 0.263 (exact 0.004) | 0.259 (exact 0.002) | 0.204 (exact 0) |
| Unigram: always the most frequent training target token | 0.108 | 0.111 | 0.124 |
| Order-blind oracle: knows the multiset and the true prefix; loss / expected token acc | 0.950 / 0.538 | 0.963 / 0.531 | 1.262 / 0.436 |

**Findings** (measured on this task, budget, and split; 3 seeds):

1. **With positional encoding in both stacks**, the model reverses in-distribution test sequences almost perfectly: exact match 0.996–1.000.
2. **Without encoder positional encoding**, exact match falls to 0.005–0.011, while the multiset of the output is correct about 95% of the time. No matched group ever has both members correct, as the symmetry in [03 §8](03_attention_math.md#8-positional-information-and-a-permutation-symmetry) requires.
   - Its teacher-forced loss (1.011 on test) and token accuracy (0.524) are close to the order-blind oracle (0.963 / 0.531).
   - So the model learns nearly what an order-blind predictor can learn, and nothing about order.
   - Its few exact successes are mostly at length 4 (`results/eval/no_enc_pos_*_test.json`, `by_length`).
3. **Teacher forcing vs. generation.**
   - The ablation's token accuracy of 0.52 coexists with exact match near 0.01: being right on about half the positions rarely yields a fully correct sequence.
   - For the baseline the two measures agree because nearly every token is right.
4. **Lengths 13–16 (distribution shift).** Neither model produces a single correct sequence. The baseline's teacher-forced loss rises to about 7 nats/token, and its outputs almost never have the correct multiset; the ablation keeps about 40%. See [05 §4](05_critical_analysis.md#4-fitting-vs-generalization) for what this does and does not imply.

**Failure examples.** Each eval JSON lists the first 5 failures in id order. For `no_enc_pos_s0` on test, examples `test-00000` and `test-00001` are a matched pair. Both receive the identical prediction `[10, 10, 4, 6, 4, 10, 7, 3, 6]`: the correct multiset, wrong for both.

**Figures** (regenerate with `python -m src.report`):
- `results/figures/curves.png`: dev loss vs. step for all 6 runs.
- `results/figures/by_length.png`: exact and multiset accuracy by length.
- `results/figures/attention.png`: one cross-attention example; see [05 §5](05_critical_analysis.md#5-attention-maps-what-can-be-read-from-them) for its limits.

## 6. What these results do not show

- Anything about translation quality, BLEU, or the paper's training cost.
- That the trends hold for other tasks, sizes, budgets, or data splits: one task, one split, and 3 initialization seeds were used.
- That sinusoidal encodings cannot extrapolate in general: only this model and budget were tested.
- Why the baseline fails on longer inputs: no probe was run.
