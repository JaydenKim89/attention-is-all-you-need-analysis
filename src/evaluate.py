"""Evaluation: teacher-forced metrics, greedy decoding, baselines.

Conventions
  * Teacher-forced loss: mean cross-entropy over non-PAD target positions (EOS included).
  * Token accuracy: argmax == target over the same non-PAD positions.
  * Exact match (autoregressive, greedy): the model must emit EOS, and the tokens
    before the first EOS must equal the reversed source. Decoding stops at EOS or
    after len(src) + 5 new tokens (a cap only; the model is not told the length).
  * Failures listed: the first 5 failed examples in id order (no selection by content).

  python -m src.evaluate --run results/runs/baseline_s0 --split test
  python -m src.evaluate --baselines --split test
"""

import argparse
import os
from collections import Counter, defaultdict

import torch

from src.common import device, load_json, save_json
from src.data import load, make_batch
from src.model import EOS, PAD, ModelConfig, Transformer, token_cross_entropy

N_FAILURES = 5
EXTRA_DECODE_STEPS = 5


@torch.no_grad()
def teacher_forced_metrics(model, examples, dev, batch_size=500):
    model.eval()
    loss_sum, n_tok, n_correct = 0.0, 0, 0
    for i in range(0, len(examples), batch_size):
        src, tgt_in, tgt_out = make_batch(examples[i:i + batch_size], dev)
        logits = model(src, tgt_in)
        valid = tgt_out != PAD
        n = int(valid.sum())
        loss_sum += token_cross_entropy(logits, tgt_out).item() * n
        n_correct += int((logits.argmax(-1)[valid] == tgt_out[valid]).sum())
        n_tok += n
    return {"loss": loss_sum / n_tok, "token_acc": n_correct / n_tok, "n_tokens": n_tok}


@torch.no_grad()
def greedy_predictions(model, examples, dev, batch_size=500):
    model.eval()
    preds = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        src, _, _ = make_batch(chunk, dev)
        outs, ended = model.greedy_decode(src, max(len(e["src"]) for e in chunk) + EXTRA_DECODE_STEPS)
        preds.extend(zip(outs, ended))
    return preds


def greedy_metrics(model, examples, dev):
    preds = greedy_predictions(model, examples, dev)
    by_len = defaultdict(lambda: [0, 0, 0])  # exact, n, multiset
    groups = defaultdict(list)
    failures = []
    n_exact = n_ended = n_len = n_multiset = 0
    for ex, (out, ended) in zip(examples, preds):
        ok = ended and out == ex["tgt"]
        n_exact += ok
        n_ended += ended
        n_len += ended and len(out) == len(ex["tgt"])
        # Right tokens in any order: separates "wrong content" from "wrong order".
        n_multiset += ended and sorted(out) == sorted(ex["tgt"])
        by_len[len(ex["src"])][0] += ok
        by_len[len(ex["src"])][1] += 1
        by_len[len(ex["src"])][2] += ended and sorted(out) == sorted(ex["tgt"])
        groups[ex["group"]].append(ok)
        if not ok and len(failures) < N_FAILURES:
            failures.append({"id": ex["id"], "src": ex["src"], "tgt": ex["tgt"], "pred": out, "emitted_eos": ended})
    both = sum(all(g) for g in groups.values())
    return {
        "exact_acc": n_exact / len(examples),
        "eos_rate": n_ended / len(examples),
        "length_match_rate": n_len / len(examples),
        "multiset_match_rate": n_multiset / len(examples),
        "groups_both_correct": both,
        "groups_one_correct": sum(sum(g) == 1 for g in groups.values()),
        "n_groups": len(groups),
        "by_length": {str(k): {"exact_acc": v[0] / v[1], "multiset_acc": v[2] / v[1], "n": v[1]} for k, v in sorted(by_len.items())},
        "failures": failures,
    }


def baselines(train, examples):
    """Copy (output = source) and unigram (most frequent training target token at every position)."""
    counts = Counter(t for e in train for t in e["tgt"] + [EOS])
    unigram = counts.most_common(1)[0][0]
    copy_correct = uni_correct = n_tok = copy_exact = 0
    for e in examples:
        target = e["tgt"] + [EOS]
        copy_pred = e["src"] + [EOS]  # same length as target, so positions align
        copy_correct += sum(a == b for a, b in zip(copy_pred, target))
        uni_correct += sum(t == unigram for t in target)
        n_tok += len(target)
        copy_exact += e["src"] == e["tgt"]
    return {"copy": {"token_acc": copy_correct / n_tok, "exact_acc": copy_exact / len(examples)},
            "unigram": {"token": unigram, "token_acc": uni_correct / n_tok, "exact_acc": 0.0},
            "n_tokens": n_tok}


def multiset_oracle(examples):
    """Reference for an order-blind predictor that knows the source multiset and the
    teacher-forced target prefix: p(next = t) = remaining count of t / remaining tokens,
    and p(EOS) = 1 once nothing remains. Returns its mean cross-entropy and the expected
    token accuracy of predicting the most frequent remaining token. This is not a model;
    it bounds what order-blind information alone can achieve under teacher forcing.
    """
    import math

    nll = acc = n = 0.0
    for e in examples:
        remaining = Counter(e["tgt"])
        left = len(e["tgt"])
        for t in e["tgt"]:
            nll -= math.log(remaining[t] / left)
            top = max(remaining.values())
            ties = sum(c == top for c in remaining.values())
            acc += 1.0 / ties if remaining[t] == top else 0.0  # argmax with uniform tie-break
            remaining[t] -= 1
            if remaining[t] == 0:
                del remaining[t]
            left -= 1
        n += len(e["tgt"]) + 1  # EOS position: probability 1, always correct
        acc += 1.0
    return {"loss": nll / n, "expected_token_acc": acc / n}


def load_run(run_dir, dev):
    cfg = load_json(os.path.join(run_dir, "config.json"))
    name = os.path.basename(os.path.normpath(run_dir))
    ckpt = torch.load(os.path.join("checkpoints", f"{name}.pt"), map_location=dev)
    model = Transformer(ModelConfig(**ckpt["model_config"])).to(dev)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg, name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="results/runs/<name>")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "test_long"])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--data-dir", default="data/reversal_v1")
    args = p.parse_args()
    os.makedirs(os.path.join("results", "eval"), exist_ok=True)
    dev = device()
    if args.baselines:
        splits = load(args.data_dir)
        out = baselines(splits["train"], splits[args.split])
        out["multiset_oracle"] = multiset_oracle(splits[args.split])
        save_json(out, os.path.join("results", "eval", f"baselines_{args.split}.json"))
        print(out)
        return
    model, cfg, name = load_run(args.run, dev)
    examples = load(cfg["data_dir"])[args.split]
    out = {"run": name, "split": args.split, **teacher_forced_metrics(model, examples, dev),
           **greedy_metrics(model, examples, dev)}
    save_json(out, os.path.join("results", "eval", f"{name}_{args.split}.json"))
    print({k: out[k] for k in ("loss", "token_acc", "exact_acc", "length_match_rate", "multiset_match_rate",
                               "groups_both_correct", "n_groups")})


if __name__ == "__main__":
    main()
