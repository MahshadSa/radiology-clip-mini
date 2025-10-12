from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import io
import json
import random
import base64
from datasets import load_dataset, Image as HFImage, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

TEXT_PRIORITY: Tuple[str, ...] = (
    "findings", "impression", "report", "text",
    "Findings", "Impression", "Report", "report_text",
)
PATIENT_KEYS: Tuple[str, ...] = ("patient_id", "uid", "subject_id", "pid", "subject", "patient")


def pick_text(rec: Dict, fields: Tuple[str, ...] = TEXT_PRIORITY) -> str:
    for k in fields:
        v = rec.get(k, "")
        if isinstance(v, str):
            v = v.strip()
            if v:
                return v
    return ""


def get_patient_id(rec: Dict) -> str:
    for k in PATIENT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    v = str(rec.get("id", "unknown"))
    return v.split("_")[0]

import binascii


def has_image(rec: Dict) -> bool:
    if "image" in rec and rec["image"] is not None:
        return True
    if "images" in rec and isinstance(rec["images"], (list, tuple)) and len(rec["images"]) > 0:
        return True
    if "image_path" in rec or "img_path" in rec:
        return True
    return False

def get_pil_image(rec: Dict) -> Image.Image:
    if "image" in rec and rec["image"] is not None:
        return rec["image"].convert("RGB")
    if "images" in rec and rec["images"]:
        return rec["images"][0].convert("RGB")
    if "image_path" in rec or "img_path" in rec:
        return Image.open(rec.get("image_path", rec.get("img_path"))).convert("RGB")
    raise KeyError("No image field found")



class IUXRayPairs(Dataset):
    def __init__(self, hf_split, image_size: int = 320):
        self.ds = hf_split
        self.tform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.keep: List[int] = []
        for i, rec in enumerate(self.ds):
    if pick_text(rec) and has_image(rec):
        try:
            _ = get_pil_image(rec)   # quick decode test
            self.keep.append(i)
        except Exception:
            pass


    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, i: int):
        rec = self.ds[self.keep[i]]
        img = get_pil_image(rec)
        txt = pick_text(rec)
        pid = get_patient_id(rec)
        img = self.tform(img)
        return {"image": img, "text": txt, "pid": pid}


def _make_patient_splits(records, max_patients: int, val_frac=0.1, test_frac=0.1, seed=42):
    rng = random.Random(seed)
    pids, seen = [], set()
    for r in records:
        if has_image(r) and pick_text(r):
            pid = get_patient_id(r)
            if pid not in seen:
                seen.add(pid)
                pids.append(pid)
    rng.shuffle(pids)
    if max_patients and max_patients > 0:
        pids = pids[:max_patients]
    n = len(pids)
    n_test = max(1, int(n * test_frac)) if n >= 3 else max(0, n - 2)
    n_val = max(1, int(n * val_frac)) if n >= 3 else 1 if n >= 2 else 0
    test_p = set(pids[:n_test])
    val_p = set(pids[n_test:n_test + n_val])
    train_p = set(pids[n_test + n_val:])
    return {"train": list(train_p), "val": list(val_p), "test": list(test_p)}


def _filter_by_pids(hf_ds, keep_pids: set):
    idx = [
        i for i, r in enumerate(hf_ds)
        if get_patient_id(r) in keep_pids and pick_text(r) and has_image(r)
    ]
    return hf_ds.select(idx)


def build_dataloaders(cfg: Dict):
    """Return dict(train=DataLoader, val=DataLoader, test=DataLoader)."""
    image_size = int(cfg["data"]["image_size"])
    num_workers = int(cfg["data"].get("num_workers", 2))
    max_patients = int(cfg["data"].get("max_patients", 50))
    split_file = cfg["data"].get("split_file", "data/splits.json")
    spec = cfg["data"].get("split_spec", "train")
    cache_dir = cfg["data"].get("cache_dir", None)

    ds = load_dataset("ykumards/open-i", split=spec, cache_dir=cache_dir)
    if "image" in ds.features:
    if not isinstance(ds.features["image"], HFImage):
        ds = ds.cast_column("image", HFImage())
elif "images" in ds.features:
    # list/sequence of images
    feat = ds.features["images"]
    if not (hasattr(feat, "feature") and isinstance(feat.feature, HFImage)):
        ds = ds.cast_column("images", Sequence(HFImage()))
    sp = Path(split_file)
    if sp.exists():
        splits = json.loads(sp.read_text())
    else:
        splits = _make_patient_splits(ds, max_patients=max_patients, seed=int(cfg["seed"]))
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(splits, indent=2))

    p_train, p_val, p_test = map(set, (splits["train"], splits["val"], splits["test"]))
    ds_train = _filter_by_pids(ds, p_train)
    ds_val = _filter_by_pids(ds, p_val)
    ds_test = _filter_by_pids(ds, p_test)

    train = IUXRayPairs(ds_train, image_size)
    val = IUXRayPairs(ds_val, image_size)
    test = IUXRayPairs(ds_test, image_size)

    def collate(batch: List[Dict]):
        imgs = torch.stack([b["image"] for b in batch])
        texts = [b["text"] for b in batch]
        return {"images": imgs, "texts": texts}

    dl_train = DataLoader(
        train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    dl_val = DataLoader(
        val,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    dl_test = DataLoader(
        test,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    return {"train": dl_train, "val": dl_val, "test": dl_test}
