from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# Text helpers 
TEXT_PRIORITY = ("findings", "impression", "report", "text")
PATIENT_KEYS = ("patient_id", "uid", "subject_id", "pid")

def pick_text(rec: Dict, fields: Tuple[str, ...] = TEXT_PRIORITY) -> str:
    for k in fields:
        v = rec.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def get_patient_id(rec: Dict) -> str:
    for k in PATIENT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: derive from image or id
    v = str(rec.get("id", "unknown"))
    return v.split("_")[0]

class IUXRayPairs(Dataset):
    def __init__(self, hf_split, image_size: int = 320):
        self.ds = hf_split
        self.tform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        # precompute indices)
        self.keep = []
        for i, rec in enumerate(self.ds):
            if rec.get("image") is None:
                continue
            if pick_text(rec):
                self.keep.append(i)

    def __len__(self): return len(self.keep)

    def __getitem__(self, i: int):
        rec = self.ds[self.keep[i]]
        img = rec["image"].convert("RGB")
        txt = pick_text(rec)
        pid = get_patient_id(rec)
        img = self.tform(img)
        return {"image": img, "text": txt, "pid": pid}

# Splits (patient-level)
def _make_patient_splits(records, max_patients: int, val_frac=0.1, test_frac=0.1, seed=42):
    rng = random.Random(seed)
    pids = []
    seen = set()
    for r in records:
        if r.get("image") is None: 
            continue
        if not pick_text(r): 
            continue
        pid = get_patient_id(r)
        if pid not in seen:
            seen.add(pid); pids.append(pid)
    rng.shuffle(pids)
    if max_patients is not None and max_patients > 0:
        pids = pids[:max_patients]
    n = len(pids)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    test_p = set(pids[:n_test])
    val_p  = set(pids[n_test:n_test+n_val])
    train_p= set(pids[n_test+n_val:])
    return {"train": list(train_p), "val": list(val_p), "test": list(test_p)}

def _filter_by_pids(hf_ds, keep_pids: set):
    idx = [i for i, r in enumerate(hf_ds) if get_patient_id(r) in keep_pids and pick_text(r) and r.get("image") is not None]
    return hf_ds.select(idx)

def build_dataloaders(cfg):
    """
    Returns: dict(train=DataLoader, val=DataLoader, test=DataLoader)
    """
    image_size = int(cfg["data"]["image_size"])
    num_workers = int(cfg["data"].get("num_workers", 2))
    max_patients = int(cfg["data"].get("max_patients", 50))
    split_file = cfg["data"].get("split_file", "data/splits.json")

    ds = load_dataset("iu_xray")["train"]  # iu_xray provides everything under 'train'
    sp = Path(split_file)
    if sp.exists():
        splits = json.loads(sp.read_text())
    else:
        splits = _make_patient_splits(ds, max_patients=max_patients, seed=int(cfg["seed"]))
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(splits, indent=2))
    p_train, p_val, p_test = map(set, (splits["train"], splits["val"], splits["test"]))

    ds_train = _filter_by_pids(ds, p_train)
    ds_val   = _filter_by_pids(ds, p_val)
    ds_test  = _filter_by_pids(ds, p_test)

    train = IUXRayPairs(ds_train, image_size)
    val   = IUXRayPairs(ds_val, image_size)
    test  = IUXRayPairs(ds_test, image_size)

    def collate(batch):
        imgs = torch.stack([b["image"] for b in batch])
        texts= [b["text"] for b in batch]
        return {"images": imgs, "texts": texts}

    dl_train = DataLoader(train, batch_size=cfg["train"]["batch_size"], shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate)
    dl_val   = DataLoader(val,   batch_size=64,                          num_workers=num_workers, pin_memory=True, collate_fn=collate)
    dl_test  = DataLoader(test,  batch_size=64,                          num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return {"train": dl_train, "val": dl_val, "test": dl_test}
