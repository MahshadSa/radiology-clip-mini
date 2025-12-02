from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import build_dataloaders
from .models import CLIPMini
from .utils import save_json


@torch.no_grad()
def embed_split(model: CLIPMini, loader, device: str):
    zs_img, zs_txt = [], []
    for b in loader:
        imgs = b["images"].to(device, non_blocking=True)
        toks = model.text_enc.tokenize(b["texts"]).to(device)
        _, zi, zt = model(imgs, toks)
        zs_img.append(zi.cpu())
        zs_txt.append(zt.cpu())
    if not zs_img:
        return None, None
    return torch.cat(zs_img, dim=0), torch.cat(zs_txt, dim=0)


def recall_at_k(sim: torch.Tensor, ks=(1, 5, 10)) -> dict:
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(sim.size(0), device=sim.device)[:, None]
    hits = (ranks[:, : max(ks)] == gt).cpu().numpy()

    out = {}
    for k in ks:
        out[f"R@{k}"] = float(hits[:, :k].any(axis=1).mean())

    rr = []
    for i in range(sim.size(0)):
        idx = (ranks[i] == i).nonzero(as_tuple=False)
        if idx.numel() == 0:
            # no correct item found, push a large rank
            rr.append(sim.size(1))
        else:
            rr.append(int(idx.item()) + 1)
    out["MedR"] = float(np.median(rr))
    return out


def ndcg_at_k(sim: torch.Tensor, k: int = 10) -> dict:
    ranks = sim.argsort(dim=1, descending=True)
    dcgs = []
    for i in range(sim.size(0)):
        idx = (ranks[i] == i).nonzero(as_tuple=False)
        if idx.numel() == 0:
            dcgs.append(0.0)
            continue
        pos = int(idx.item())
        if pos < k:
            dcg = 1.0 / np.log2(pos + 2)
        else:
            dcg = 0.0
        dcgs.append(dcg)
    return {f"nDCG@{k}": float(np.mean(dcgs))}


def main(cfg_path: str, ckpt_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dls = build_dataloaders(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPMini(
        embed_dim=cfg["model"]["embed_dim"],
        text_model=cfg["model"]["text_encoder"],
        tau_init=cfg["model"]["temperature_init"],
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    zi_val, zt_val = embed_split(model, dls["val"], device)
    if zi_val is None or zt_val is None:
        raise RuntimeError("Validation loader produced no embeddings")

    sim = zi_val @ zt_val.t()

    it = recall_at_k(sim)
    it.update(ndcg_at_k(sim))

    ti = recall_at_k(sim.t())
    ti.update(ndcg_at_k(sim.t()))

    metrics = {
        "image_to_text": it,
        "text_to_image": ti,
    }

    out_dir = Path(ckpt_path).parent
    save_json(metrics, out_dir / "metrics.json")
    print("I→T", metrics["image_to_text"])
    print("T→I", metrics["text_to_image"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
