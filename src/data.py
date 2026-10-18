"""Synthetic sequence-reversal data with permutation-matched groups.

Every example belongs to a group of two distinct source sequences that have the
same length and the same token multiset. Groups, not examples, are assigned to
splits. Without positional information in the encoder, the two sources of a group
produce the same encoder output set (see paper_notes/03_attention_math.md, Sec. 6),
so the task is ambiguous for that model by construction.

Token ids: 0 = PAD, 1 = BOS, 2 = EOS, 3 .. 3 + n_symbols - 1 = content symbols.

  python -m src.data --out data/reversal_v1 --seed 1234
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

from src.model import BOS, EOS, PAD

DEFAULTS = {
    "seed": 1234,
    "n_symbols": 8,
    "group_size": 2,
    "train_groups": 4000,
    "dev_groups": 500,
    "test_groups": 500,
    "length_range": [4, 12],
    "long_test_groups": 250,
    "long_length_range": [13, 16],
}


def _make_group(rng, length, n_symbols, group_size, seen):
    """Return group_size distinct, previously unseen orderings of one random multiset."""
    for _ in range(1000):
        multiset = [rng.randrange(n_symbols) + 3 for _ in range(length)]
        if len(set(multiset)) < 2:
            continue
        orders = []
        for _ in range(100):
            perm = multiset[:]
            rng.shuffle(perm)
            t = tuple(perm)
            if t not in seen and t not in orders:
                orders.append(t)
            if len(orders) == group_size:
                seen.update(orders)
                return [list(o) for o in orders]
    raise RuntimeError("could not build a group; increase n_symbols or length")


def generate(cfg):
    """Deterministic dataset from cfg; returns {split: [example, ...]}."""
    rng = random.Random(cfg["seed"])
    seen = set()  # global over all splits: no source appears twice anywhere
    splits = {}
    plan = [("train", cfg["train_groups"], cfg["length_range"]),
            ("dev", cfg["dev_groups"], cfg["length_range"]),
            ("test", cfg["test_groups"], cfg["length_range"]),
            ("test_long", cfg["long_test_groups"], cfg["long_length_range"])]
    gid = 0
    for name, n_groups, (lo, hi) in plan:
        examples = []
        for _ in range(n_groups):
            length = rng.randint(lo, hi)
            for src in _make_group(rng, length, cfg["n_symbols"], cfg["group_size"], seen):
                examples.append({"id": f"{name}-{len(examples):05d}", "group": gid,
                                 "src": src, "tgt": src[::-1]})
            gid += 1
        splits[name] = examples
    return splits


def validate(splits, group_size=2):
    """Raise AssertionError if split construction is wrong. Returns summary counts."""
    group_split, sources = {}, {}
    for name, examples in splits.items():
        for ex in examples:
            assert ex["tgt"] == ex["src"][::-1], f"{ex['id']}: target is not the reversed source"
            assert all(t >= 3 for t in ex["src"]), f"{ex['id']}: special token in source"
            assert group_split.setdefault(ex["group"], name) == name, f"group {ex['group']} spans splits"
            key = tuple(ex["src"])
            assert key not in sources, f"duplicate source {key} in {sources.get(key)} and {name}"
            sources[key] = name
    groups = {}
    for examples in splits.values():
        for ex in examples:
            groups.setdefault(ex["group"], []).append(ex["src"])
    for g, members in groups.items():
        assert len(members) == group_size, f"group {g} has {len(members)} members"
        assert len({len(m) for m in members}) == 1, f"group {g}: lengths differ"
        assert len({tuple(sorted(m)) for m in members}) == 1, f"group {g}: multisets differ"
        assert len({tuple(m) for m in members}) == group_size, f"group {g}: orders not distinct"
    return {name: len(ex) for name, ex in splits.items()}


def make_batch(examples, device="cpu"):
    """Pad a list of examples into (src, tgt_in, tgt_out) LongTensors.

    tgt_in = [BOS] + y and tgt_out = y + [EOS] (teacher forcing, shifted by one);
    both are right-padded with PAD, which the loss and masks ignore.
    """
    import torch

    Ls = max(len(e["src"]) for e in examples)
    Lt = max(len(e["tgt"]) for e in examples) + 1
    src = torch.full((len(examples), Ls), PAD, dtype=torch.long)
    tgt_in = torch.full((len(examples), Lt), PAD, dtype=torch.long)
    tgt_out = torch.full((len(examples), Lt), PAD, dtype=torch.long)
    for i, e in enumerate(examples):
        src[i, : len(e["src"])] = torch.tensor(e["src"])
        tgt_in[i, : len(e["tgt"]) + 1] = torch.tensor([BOS] + e["tgt"])
        tgt_out[i, : len(e["tgt"]) + 1] = torch.tensor(e["tgt"] + [EOS])
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


def load(data_dir):
    splits = {}
    for name in ("train", "dev", "test", "test_long"):
        with open(os.path.join(data_dir, f"{name}.jsonl")) as f:
            splits[name] = [json.loads(line) for line in f]
    return splits


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/reversal_v1")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--manifest", default="results/dataset_manifest.json")
    args = p.parse_args()
    cfg = dict(DEFAULTS, seed=args.seed)
    splits = generate(cfg)
    counts = validate(splits, cfg["group_size"])
    os.makedirs(args.out, exist_ok=True)
    files = {}
    for name, examples in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", newline="\n") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        files[name] = {"examples": counts[name], "sha256": _sha256(path),
                       "lengths": dict(sorted(Counter(len(e["src"]) for e in examples).items()))}
    manifest = {"generator": "src/data.py", "config": cfg, "files": files,
                "vocab": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "symbols": [3, 3 + cfg["n_symbols"] - 1]}}
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v["examples"] for k, v in files.items()}))


if __name__ == "__main__":
    main()
