from __future__ import annotations
import json, random, subprocess, time, os
from pathlib import Path
from typing import Any
import numpy as np
import torch


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(root: str = "results") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = Path(root) / ts
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_latest(ckpt_path: Path, root: str = "results"):
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "latest.txt").write_text(str(ckpt_path) + "\n", encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)
