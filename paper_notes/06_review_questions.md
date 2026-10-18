# Open Review Questions

Questions tied to the artifacts in this repository. They are for the repository owner to work through without reading the implementation first. The implementation and these notes were produced with AI assistance, so working through them is how the owner's own understanding gets checked. Nothing here records that any check has been passed.

| # | Question | Artifact to compare against afterwards | Owner review |
|---|---|---|---|
| 1 | Derive ∂L/∂Q, ∂L/∂K, ∂L/∂V for O = softmax(QKᵀ/√d_k)V, given G = ∂L/∂O. | `paper_notes/03_attention_math.md` §6, `src/attention_numpy.py::attention_backward` | pending |
| 2 | Which assumptions make Var(q·k) = d_k, and which of them fail for a trained model? | §2 of the same note; paper footnote 4 | pending |
| 3 | For the example `src = [3, 4, 5]`, write `tgt_in`, `tgt_out`, the source mask, and the causal target mask. | `tests/test_model.py::test_target_shift_and_eos`, `src/model.py::tgt_mask` | pending |
| 4 | Show why removing encoder positional encoding makes the two members of a matched group indistinguishable to greedy decoding, but not under teacher forcing. | `03_attention_math.md` §8; `groups_both_correct` in `results/eval/no_enc_pos_*` | pending |
| 5 | Explain the gap between the ablation's teacher-forced token accuracy (about 0.52) and its exact-sequence accuracy (about 0.01). | `results/summary.csv`, `multiset_oracle` in `src/evaluate.py` | pending |
| 6 | The 25 tests pass. What do they *not* establish about learning or generalization? | `paper_notes/05_critical_analysis.md` §2 | pending |
| 7 | Why does the baseline fail on lengths 13–16, and what experiment would test your explanation? | `results/eval/baseline_*_test_long.json` | pending |
| 8 | In the worked example with a causal mask, why are rows 0 and 1 of ∂L/∂Q exactly zero? | `python -m src.worked_example` | pending |
