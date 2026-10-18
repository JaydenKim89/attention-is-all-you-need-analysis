# Understanding Transformers Through Derivation, Implementation, and Controlled Experiments

A study of *Attention Is All You Need* (Vaswani et al., NeurIPS 2017; [arXiv 1706.03762](https://arxiv.org/abs/1706.03762), v7 used here). It works through the attention mathematics, builds a small encoder-decoder Transformer, and runs one controlled experiment on a synthetic sequence-reversal task.

**Question:** can the mathematics, the implementation's behavior, and the learning results of a small Transformer be explained and checked consistently, including where the model fails?

**Scope:** sequence modeling, attention, positional information as an inductive bias, gradient-based learning, numerical correctness, experimental controls, and generalization *within this toy setting*. This is not a reproduction of the paper's translation results.

## Status (started September 6, 2026)

| Part | State |
|---|---|
| Attention derivations: forward, backward, masks, scaling assumptions, permutation symmetry | Written: [paper_notes/03](paper_notes/03_attention_math.md) |
| NumPy attention reference with manual backward pass | Implemented; checked against finite differences and PyTorch autograd |
| Small encoder-decoder Transformer (PyTorch tensor ops; no `nn.Transformer` / `nn.MultiheadAttention`) | Implemented; 25 unit and functional tests pass |
| Reversal experiment: baseline vs. no encoder positional encoding, 3 seeds each | Run once with frozen configs; results in [paper_notes/04](paper_notes/04_training_and_results.md) |
| Longer-length test (13–16, distribution shift) | Run; both models fail |
| Not done | Beam search, other ablations (scale, heads, label smoothing), any translation experiment |

## Findings (measured on this task and budget; 3 seeds)

- **Position matters exactly where the symmetry says it should.** With positional encoding in both stacks, test exact-sequence accuracy is 0.996–1.000. Removing it from the encoder only drops exact accuracy to 0.005–0.011.
  - The ablated model still outputs the right multiset of tokens about 95% of the time.
  - Its teacher-forced loss (1.01 nats/token) is close to the best achievable by any order-blind predictor (0.96).
  - Matched pairs of permuted inputs never both succeed, as the permutation-invariance argument predicts ([03 §8](paper_notes/03_attention_math.md#8-positional-information-and-a-permutation-symmetry)).
- **Neither model generalizes to longer inputs.** On lengths 13–16, unseen in training, no sequence is reversed correctly. The baseline's teacher-forced loss jumps to about 7 nats/token.

Limits and interpretation: [paper_notes/05](paper_notes/05_critical_analysis.md).

## Repository map

```
paper_notes/   01 motivation · 02 architecture + implementation map · 03 attention math ·
               04 training and results · 05 critical analysis · 06 open review questions
src/           attention_numpy.py (reference + manual backward), worked_example.py,
               model.py, data.py, train.py, evaluate.py, report.py, common.py
tests/         test_attention.py (numerical, gradients), test_model.py (masks, shift, symmetry, splits)
configs/       baseline.json, no_enc_pos.json (differ only in enc_pos)
scripts/       reproduce.sh (end to end)
results/       dataset_manifest.json, runs/*/{config,history,summary}, eval/*.json,
               summary.csv, figures/*.png, attention_example.json
```

Each claim can be traced as: derivation (`paper_notes/03`) or configuration (`configs/`) → implementation (`src/`) → test (`tests/`) or run record (`results/runs/`) → measured artifact (`results/eval/`, `results/summary.csv`) → interpretation (`paper_notes/04–05`).

## Reproducing

The recorded environment is Python 3.14.5, NumPy 2.5.3, PyTorch 2.14.1+cu130, matplotlib 3.11.2, on Windows 11 with one RTX 5070 Ti. Each training run took about 1 minute. Install with `pip install -r requirements.txt`, choosing the PyTorch build for your platform.

Everything, from the repository root:

```bash
bash scripts/reproduce.sh
```

Or step by step:

```bash
python -m unittest discover -s tests -t .
```

```bash
python -m src.worked_example
```

```bash
python -m src.data --out data/reversal_v1 --seed 1234
```

```bash
python -m src.train --config configs/baseline.json --seed 0 --subset 64 --steps 600 --name overfit_check_64
```

```bash
python -m src.train --config configs/no_enc_pos.json --seed 0
```

```bash
python -m src.evaluate --run results/runs/no_enc_pos_s0 --split test
```

```bash
python -m src.report
```

Generated data (`data/`) and checkpoints (`checkpoints/`) are not committed; `scripts/reproduce.sh` recreates them. Every run's `summary.json` records the git commit, whether the tree was dirty, a SHA-256 of `src/*.py` and `configs/*.json`, the seeds, and the environment.

## Boundaries

- A 232,640-parameter model on an 8-symbol reversal task, trained for 3,000 steps, says nothing about translation quality, BLEU, or the paper's training cost.
- Results come from one data split and three initialization seeds.
- The implementation and notes were written with AI assistance. [paper_notes/06](paper_notes/06_review_questions.md) lists questions for the owner to work through independently; none is marked as completed.
