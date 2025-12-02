from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_dir(root: str = "results") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = Path(root) / ts
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_latest(ckpt_path: Path, root: str = "results") -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "latest.txt").write_text(str(ckpt_path))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
