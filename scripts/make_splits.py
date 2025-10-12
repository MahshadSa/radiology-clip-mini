from __future__ import annotations
import json, random
from pathlib import Path
from datasets import load_dataset

from rclip.data import pick_text, get_patient_id

def main(out_path="data/splits.json", max_patients=50, val_frac=0.1, test_frac=0.1, seed=42):
    ds = load_dataset("iu_xray")["train"]
    rng = random.Random(seed)
    pids = []
    seen = set()
    for r in ds:
        if r.get("image") is None: 
            continue
        if not pick_text(r): 
            continue
        pid = get_patient_id(r)
        if pid not in seen:
            seen.add(pid); pids.append(pid)
    rng.shuffle(pids)
    if max_patients and max_patients > 0:
        pids = pids[:max_patients]
    n = len(pids)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    splits = {
        "train": pids[n_test+n_val:],
        "val":   pids[n_test:n_test+n_val],
        "test":  pids[:n_test],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(splits, indent=2))
    print(f"wrote {out_path} ({len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} pids)")

if __name__ == "__main__":
    main()
