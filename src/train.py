"""Train the small Transformer on the reversal task with teacher forcing.

  python -m src.train --config configs/baseline.json --seed 0
  python -m src.train --config configs/baseline.json --seed 0 --subset 64 --name overfit_check

Writes results/runs/<name>/{config.json, history.csv, summary.json} and a
checkpoint to checkpoints/<name>.pt (not committed). Only train and dev data are
used here; the test splits are read by src/evaluate.py after configs are frozen.
"""

import argparse
import csv
import os
import time

import torch

from src.common import device, load_json, provenance, save_json, set_seed
from src.data import load, make_batch
from src.evaluate import greedy_metrics, teacher_forced_metrics
from src.model import ModelConfig, Transformer, count_parameters, token_cross_entropy


def noam_lr(step, d_model, warmup, scale):
    """Eq. (3): d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), times a project scale factor."""
    step = max(step, 1)
    return scale * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def batches(examples, batch_size, generator):
    """Endless stream of shuffled mini-batches (reshuffled every epoch)."""
    while True:
        order = torch.randperm(len(examples), generator=generator).tolist()
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [examples[j] for j in order[i:i + batch_size]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--subset", type=int, default=0, help="train and evaluate on the first N train examples")
    p.add_argument("--steps", type=int, default=0, help="override the configured step budget")
    args = p.parse_args()

    cfg = load_json(args.config)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    name = args.name or f"{cfg['name']}_s{args.seed}"
    out_dir = os.path.join("results", "runs", name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    set_seed(args.seed)
    dev = device()
    splits = load(cfg["data_dir"])
    train = splits["train"][: args.subset] if args.subset else splits["train"]
    dev_set = train if args.subset else splits["dev"]

    mcfg = ModelConfig(**cfg["model"])
    model = Transformer(mcfg).to(dev)
    oc, tc = cfg["optim"], cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=tuple(oc["betas"]), eps=oc["eps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: noam_lr(s + 1, mcfg.d_model, oc["warmup_steps"], oc["lr_scale"]))
    gen = torch.Generator().manual_seed(args.seed)
    stream = batches(train, min(tc["batch_size"], len(train)), gen)

    history, running, n_running, t0 = [], 0.0, 0, time.time()
    for step in range(1, tc["steps"] + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(next(stream), dev)
        loss = token_cross_entropy(model(src, tgt_in), tgt_out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        running, n_running = running + loss.item(), n_running + 1
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            tf = teacher_forced_metrics(model, dev_set, dev)
            history.append({"step": step, "lr": sched.get_last_lr()[0], "train_loss": running / n_running,
                            "dev_loss": tf["loss"], "dev_token_acc": tf["token_acc"]})
            print(f"step {step:6d}  lr {history[-1]['lr']:.2e}  train {history[-1]['train_loss']:.4f}  "
                  f"eval {tf['loss']:.4f}  eval_tok {tf['token_acc']:.4f}", flush=True)
            running, n_running = 0.0, 0

    elapsed = time.time() - t0
    torch.save({"model": model.state_dict(), "model_config": mcfg.to_dict()}, os.path.join("checkpoints", f"{name}.pt"))
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    final_dev = {**teacher_forced_metrics(model, dev_set, dev), **greedy_metrics(model, dev_set, dev)}
    final_dev.pop("by_length", None)
    final_dev.pop("failures", None)
    save_json(cfg, os.path.join(out_dir, "config.json"))
    save_json({"name": name, "seed": args.seed, "subset": args.subset, "eval_split": "train_subset" if args.subset else "dev",
               "parameters": count_parameters(model), "precision": "float32 (TF32 disabled)",
               "train_seconds": round(elapsed, 1), "final_eval": final_dev,
               "dataset_manifest": "results/dataset_manifest.json", "provenance": provenance()},
              os.path.join(out_dir, "summary.json"))
    print({k: final_dev[k] for k in ("loss", "token_acc", "exact_acc")})


if __name__ == "__main__":
    main()
