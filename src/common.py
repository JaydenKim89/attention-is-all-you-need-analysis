"""Shared helpers: config loading, seeding, run provenance."""

import json
import platform
import subprocess
import sys

import torch


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", newline="\n") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    # TF32 off so float32 matmuls are not silently reduced in precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def source_sha256(paths=None):
    """SHA-256 over src/*.py and configs/*.json (sorted), so a run can be matched to
    exact source files even when the working tree is not committed."""
    import glob
    import hashlib

    h = hashlib.sha256()
    for path in sorted(paths or glob.glob("src/*.py") + glob.glob("configs/*.json")):
        with open(path, "rb") as f:
            h.update(path.replace("\\", "/").encode() + b"\0" + f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def provenance():
    """Source revision and environment actually used for a run."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_sha256(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "platform": platform.platform(),
    }
