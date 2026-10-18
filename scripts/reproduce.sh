#!/usr/bin/env bash
# Regenerate every committed result from scratch. Run from the repository root.
# Observed on one RTX 5070 Ti: about 1 minute per training run, 7 runs in total.
set -euo pipefail

python -m unittest discover -s tests -t .                      # numerical + functional tests
python -m src.data --out data/reversal_v1 --seed 1234           # dataset + results/dataset_manifest.json

# Tiny overfit check: baseline config on the first 64 training examples.
python -m src.train --config configs/baseline.json --seed 0 --subset 64 --steps 600 --name overfit_check_64

# Frozen configs (chosen on dev only), 3 initialization seeds each.
for cfg in baseline no_enc_pos; do
  for seed in 0 1 2; do
    python -m src.train --config "configs/$cfg.json" --seed "$seed"
  done
done

# Evaluation: dev, test (in-distribution), test_long (length shift, 13-16).
for split in dev test test_long; do
  python -m src.evaluate --baselines --split "$split"
  for cfg in baseline no_enc_pos; do
    for seed in 0 1 2; do
      python -m src.evaluate --run "results/runs/${cfg}_s${seed}" --split "$split"
    done
  done
done

python -m src.report --dump-attention results/runs/baseline_s0  # summary.csv + figures
