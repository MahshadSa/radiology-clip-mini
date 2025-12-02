from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


TEXT_PRIORITY: Tuple[str, ...] = (
    "findings",
    "impression",
    "report",
    "text",
    "Findings",
    "Impression",
    "Report",
    "report_text",
)
PATIENT_KEYS: Tuple[str, ...] = (
    "patient_id",
    "uid",
    "subject_id",
    "pid",
    "subject",
    "patient",
)


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
    v = str(rec.get("uid", rec.get("id", "unknown")))
    return v.split("_")[0]


def has_image(rec: Dict) -> bool:
    # For ykumards/open-i the actual pixels are in img_frontal / img_lateral
    if rec.get("img_frontal") is not None:
        return True
    if rec.get("img_lateral") is not None:
        return True
    if "image_path" in rec or "img_path" in rec:
        return True
    return False


def _bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def get_pil_image(rec: Dict) -> Image.Image:
    if rec.get("img_frontal") is not None:
        return _bytes_to_pil(rec["img_frontal"])
    if rec.get("img_lateral") is not None:
        return _bytes_to_pil(rec["img_lateral"])
    if "image_path" in rec:
        return Image.open(rec["image_path"]).convert("RGB")
    if "img_path" in rec:
        return Image.open(rec["img_path"]).convert("RGB")
    raise KeyError("No usable image field found")


class IUXRayPairs(Dataset):
    def __init__(self, hf_split, image_size: int = 320):
        self.ds = hf_split
        self.tform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.keep: List[int] = []
        for i, rec in enumerate(self.ds):
            if pick_text(rec) and has_image(rec):
                try:
                    _ = get_pil_image(rec)
                except Exception:
                    continue
                self.keep.append(i)

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, i: int):
        rec = self.ds[self.keep[i]]
        img = get_pil_image(rec)
        txt = pick_text(rec)
        pid = get_patient_id(rec)
        img = self.tform(img)
        return {"image": img, "text": txt, "pid": pid}


def _make_patient_splits(
    records,
    max_patients: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
):
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
    if n == 0:
        return {"train": [], "val": [], "test": []}
    n_test = max(1, int(n * test_frac)) if n >= 3 else max(0, n - 2)
    n_val = max(1, int(n * val_frac)) if n >= 3 else (1 if n >= 2 else 0)
    test_p = set(pids[:n_test])
    val_p = set(pids[n_test : n_test + n_val])
    train_p = set(pids[n_test + n_val :])
    return {"train": list(train_p), "val": list(val_p), "test": list(test_p)}


def _filter_by_pids(hf_ds, keep_pids: set):
    idx = [
        i
        for i, r in enumerate(hf_ds)
        if get_patient_id(r) in keep_pids and pick_text(r) and has_image(r)
    ]
    return hf_ds.select(idx)


def collate_batch(batch: List[Dict]):
    imgs = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"images": imgs, "texts": texts}


def build_dataloaders(cfg: Dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    image_size = int(data_cfg.get("image_size", 320))
    num_workers = int(data_cfg.get("num_workers", 2))
    max_patients = int(data_cfg.get("max_patients", 50))
    split_file = data_cfg.get("split_file", "data/splits.json")
    spec = data_cfg.get("split_spec", "train")
    cache_dir = data_cfg.get("cache_dir", None)

    ds = load_dataset("ykumards/open-i", split=spec, cache_dir=cache_dir)

    sp = Path(split_file)
    if sp.exists():
        splits = json.loads(sp.read_text())
    else:
        splits = _make_patient_splits(ds, max_patients=max_patients, seed=int(cfg["seed"]))
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(splits, indent=2))

    p_train = set(splits.get("train", []))
    p_val = set(splits.get("val", []))
    p_test = set(splits.get("test", []))

    ds_train = _filter_by_pids(ds, p_train)
    ds_val = _filter_by_pids(ds, p_val)
    ds_test = _filter_by_pids(ds, p_test)

    train_ds = IUXRayPairs(ds_train, image_size)
    val_ds = IUXRayPairs(ds_val, image_size)
    test_ds = IUXRayPairs(ds_test, image_size)

    pin = torch.cuda.is_available()
    batch_size = int(train_cfg.get("batch_size", 16))
    eval_batch_size = int(data_cfg.get("eval_batch_size", 64))

    dl_train = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    dl_test = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_batch,
    )
    return {"train": dl_train, "val": dl_val, "test": dl_test}
