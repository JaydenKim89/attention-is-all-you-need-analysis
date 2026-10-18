"""Summary table and figures from committed run/eval artifacts.

  python -m src.report                       # table + figures from results/
  python -m src.report --dump-attention results/runs/baseline_s0   # needs the checkpoint

Outputs: results/summary.csv, results/figures/{curves,by_length,attention}.png.
The attention figure is drawn from results/attention_example.json, so it can be
regenerated without the checkpoint once that file exists.
"""

import argparse
import csv
import glob
import json
import os
from statistics import mean

import torch

RUN_GROUPS = ("baseline", "no_enc_pos")
SPLITS = ("dev", "test", "test_long")
METRICS = ("loss", "token_acc", "exact_acc", "length_match_rate", "multiset_match_rate")


def load_evals():
    evals = {}
    for path in glob.glob(os.path.join("results", "eval", "*_s[0-9]_*.json")):
        with open(path) as f:
            e = json.load(f)
        evals[(e["run"], e["split"])] = e
    return evals


def summary_table(evals):
    rows = []
    for group in RUN_GROUPS:
        for split in SPLITS:
            runs = [e for (run, s), e in evals.items() if s == split and run.rsplit("_s", 1)[0] == group]
            if not runs:
                continue
            row = {"config": group, "split": split, "seeds": len(runs)}
            for m in METRICS:
                vals = [r[m] for r in runs]
                row[f"{m}_mean"], row[f"{m}_min"], row[f"{m}_max"] = mean(vals), min(vals), max(vals)
            row["groups_both_correct_max"] = max(r["groups_both_correct"] for r in runs)
            rows.append(row)
    for split in SPLITS:
        path = os.path.join("results", "eval", f"baselines_{split}.json")
        if os.path.exists(path):
            with open(path) as f:
                b = json.load(f)
            for name in ("copy", "unigram"):
                rows.append({"config": f"{name}_baseline", "split": split, "seeds": 0,
                             "token_acc_mean": b[name]["token_acc"], "exact_acc_mean": b[name]["exact_acc"]})
            if "multiset_oracle" in b:  # order-blind reference (teacher forced); not a trained model
                rows.append({"config": "multiset_oracle", "split": split, "seeds": 0,
                             "loss_mean": b["multiset_oracle"]["loss"],
                             "token_acc_mean": b["multiset_oracle"]["expected_token_acc"]})
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(os.path.join("results", "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_curves(plt):
    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(glob.glob(os.path.join("results", "runs", "*_s[0-9]", "history.csv"))):
        run = os.path.basename(os.path.dirname(path))
        group = run.rsplit("_s", 1)[0]
        with open(path) as f:
            h = list(csv.DictReader(f))
        color = "tab:blue" if group == "baseline" else "tab:orange"
        ax.plot([int(r["step"]) for r in h], [float(r["dev_loss"]) for r in h], color=color, alpha=0.7,
                label=group if run.endswith("_s0") else None)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dev cross-entropy (nats/token, PAD excluded)")
    ax.set_title("Teacher-forced dev loss, 3 seeds per config", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("results", "figures", "curves.png"), dpi=150)
    plt.close(fig)


def plot_by_length(plt, evals):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric, title in ((axes[0], "exact_acc", "Exact sequence accuracy (greedy)"),
                              (axes[1], None, "Correct multiset, any order (greedy)")):
        for group, color in (("baseline", "tab:blue"), ("no_enc_pos", "tab:orange")):
            for split, marker in (("test", "o"), ("test_long", "s")):
                runs = [e for (run, s), e in evals.items() if s == split and run.rsplit("_s", 1)[0] == group]
                if not runs:
                    continue
                lengths = sorted(int(k) for k in runs[0]["by_length"])
                key = metric or "multiset_acc"
                ys = [mean(r["by_length"][str(L)].get(key, 0.0) for r in runs) for L in lengths]
                ax.plot(lengths, ys, marker=marker, color=color, label=f"{group} ({split})")
        ax.axvline(12.5, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Source length (train lengths 4-12; 13-16 = held-out shift)")
        ax.set_title(title + ", mean of 3 seeds", fontsize=9)
    axes[0].set_ylabel("Fraction of examples")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "figures", "by_length.png"), dpi=150)
    plt.close(fig)


def dump_attention(run_dir):
    """Cross-attention of the last decoder layer for the first test example (teacher forced)."""
    from src.common import device
    from src.data import load, make_batch
    from src.evaluate import load_run

    dev = device()
    model, cfg, name = load_run(run_dir, dev)
    ex = load(cfg["data_dir"])["test"][0]
    layer = model.dec_layers[-1].cross_attn
    layer.record = True
    src, tgt_in, _ = make_batch([ex], dev)
    with torch.no_grad():
        model(src, tgt_in)
    out = {"run": name, "example": ex, "layer": "decoder layer %d cross-attention" % len(model.dec_layers),
           "probs": layer.last_attn[0].cpu().tolist()}  # (heads, target positions, source positions)
    with open(os.path.join("results", "attention_example.json"), "w", newline="\n") as f:
        json.dump(out, f)


def plot_attention(plt):
    path = os.path.join("results", "attention_example.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        a = json.load(f)
    heads = a["probs"]
    fig, axes = plt.subplots(1, len(heads), figsize=(3 * len(heads), 3.2))
    src = [str(t) for t in a["example"]["src"]]
    tgt_in = ["BOS"] + [str(t) for t in a["example"]["tgt"]]
    for i, (ax, p) in enumerate(zip(axes, heads)):
        ax.imshow(p, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(src)), src, fontsize=7)
        ax.set_yticks(range(len(tgt_in)), tgt_in, fontsize=7)
        ax.set_title(f"head {i}", fontsize=9)
        ax.set_xlabel("source token (position order)", fontsize=8)
    axes[0].set_ylabel("decoder input token", fontsize=8)
    fig.suptitle(f"{a['run']}, {a['layer']}, test example {a['example']['id']} (teacher forced)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "figures", "attention.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-attention", metavar="RUN_DIR")
    args = p.parse_args()
    if args.dump_attention:
        dump_attention(args.dump_attention)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    evals = load_evals()
    rows = summary_table(evals)
    plot_curves(plt)
    plot_by_length(plt, evals)
    plot_attention(plt)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("config", "split", "seeds", "token_acc_mean", "exact_acc_mean", "multiset_match_rate_mean")})


if __name__ == "__main__":
    main()
