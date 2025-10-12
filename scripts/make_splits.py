from __future__ import annotations
import json, random
from pathlib import Path
from datasets import load_dataset

from rclip.data import pick_text, get_patient_id

def main(out_path="data/splits.json", max_patients=50, val_frac=0.1, test_frac=0.1, seed=42,
         split_spec="train", cache_dir=None):
    ds = load_dataset("ykumards/open-i", split=split_spec, cache_dir=cache_dir)
             
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--max-patients", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-spec", default="train[:500]")
    ap.add_argument("--cache-dir", default="data/hf")
    args = ap.parse_args()
    main(args.out, args.max_patients, args.val_frac, args.test_frac, args.seed,
         split_spec=args.split_spec, cache_dir=args.cache_dir)
