from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

PATIENT_ID_KEYS = ("patient_id", "uid", "subject_id", "pid")

def pick_text(
    record: Dict,
    preferred_fields: Tuple[str, ...] = ("findings", "impression", "report"),
) -> str:
    """Return the first non-empty text field, in priority order."""
    for key in preferred_fields:
        value = record.get(key, "")
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    value = record.get("text", "")
    return value.strip() if isinstance(value, str) else ""


def derive_patient_id(record: Dict) -> str:
    """
    Get patient/group ID.

    Priority:
      1) Known patient-id keys.
      2) Extract a numeric chunk.
      3) Checks for "uid" or "id".
      4) Last-resort: a hash over the record's items.
    """
    for key in PATIENT_ID_KEYS:
        value = record.get(key)
        if value:
            return str(value)

    for key in ("image_id", "image_path", "path", "image"):
        if key in record and record[key]:
            s = str(record[key])
            m = re.search(r"(\d{3,})", s)  # first long numeric run
            if m:
                return m.group(1)

    for key in ("uid", "id"):
        value = record.get(key)
        if value:
            s = str(value)
            m = re.search(r"(\d{3,})", s)
            return m.group(1) if m else s

    # If no usable ID is found
    try:
        stable = json.dumps(record, sort_keys=True)
        return str(abs(hash(stable)))
    except Exception:
        return str(abs(hash(tuple(sorted(record.items())))))


def concat_all_splits(ds: DatasetDict) -> Dataset:
    """Concatenate train/validation/test splits if multiple exist."""
    parts = list(ds.values())
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def subsample_by_patient(
    indices: np.ndarray,
    patient_ids: List[str],
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    patients = np.array([patient_ids[i] for i in indices])
    unique_patients = np.unique(patients)
    rng.shuffle(unique_patients)

    kept: List[int] = []
    for p in unique_patients:
        kept.extend(indices[patients == p].tolist())
        if len(kept) >= max_samples:
            break
    return np.array(kept[:max_samples], dtype=int)


def load_iu_xray_flat(
    preferred_fields: Tuple[str, ...] = ("findings", "impression", "report")) -> Dict[str, List]:

    ds: DatasetDict = load_dataset("iu_xray")
    full: Dataset = concat_all_splits(ds)

    images, texts, patient_ids = [], [], []
    for rec in full:
        txt = pick_text(rec, preferred_fields)
        img = rec.get("image", None)  # HF returns PIL images 
        if img is None or not txt:
            continue
        pid = derive_patient_id(rec)
        images.append(img)
        texts.append(txt)
        patient_ids.append(pid)

    return {"image": images, "text": texts, "patient_id": patient_ids}


def create_patient_splits(
    n_examples: int,
    patient_ids: List[str],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Make patient-level splits and return absolute index lists.
    """
    assert 0 < val_frac < 0.5 and 0 < test_frac < 0.5 and (val_frac + test_frac) < 0.9

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_examples)
    pids = np.array(patient_ids)
    unique_patients = np.unique(pids)
    rng.shuffle(unique_patients)

    n_val = max(1, int(round(len(unique_patients) * val_frac)))
    n_test = max(1, int(round(len(unique_patients) * test_frac)))
    n_train = max(1, len(unique_patients) - n_val - n_test)

    train_pat = set(unique_patients[:n_train])
    val_pat   = set(unique_patients[n_train:n_train + n_val])
    test_pat  = set(unique_patients[n_train + n_val:])

    train_idx = all_idx[np.isin(pids, list(train_pat))].tolist()
    val_idx   = all_idx[np.isin(pids, list(val_pat))].tolist()
    test_idx  = all_idx[np.isin(pids, list(test_pat))].tolist()

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_indices_with_subsample(
    patient_ids: List[str],
    max_samples: int = 50,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(patient_ids))
    sub_indices = subsample_by_patient(all_indices, patient_ids, max_samples, rng)

    # Create splits within the subset 
    sub_pids = [patient_ids[i] for i in sub_indices]
    rel_splits = create_patient_splits(len(sub_indices), sub_pids, val_frac, test_frac, seed)

    # Map back to absolute indices
    abs_splits = {k: [int(sub_indices[i]) for i in rel_idx] for k, rel_idx in rel_splits.items()}
    return abs_splits


def save_splits_json(splits: Dict[str, List[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)



