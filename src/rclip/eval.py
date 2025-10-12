from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
import numpy as np
import torch
from .data import build_dataloaders
from .models import CLIPMini
from .utils import save_json

@torch.no_grad()
def embed_split(model, loader, device):
    zs_img, zs_txt = [], []
    for b in loader:
        imgs = b["images"].to(device, non_blocking=True)
        toks = model.text_enc.tokenize(b["texts"]).to(device)
        _, zi, zt = model(imgs, toks)
        zs_img.append(zi.cpu()); zs_txt.append(zt.cpu())
    return torch.cat(zs_img), torch.cat(zs_txt)

def recall_at_k(sim, ks=(1,5,10)):
    # rows = queries, cols = candidates
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(sim.size(0))[:, None]
    hits = (ranks[:, : max(ks)] == gt).cpu().numpy()
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(hits[:, :k].any(axis=1).mean())
    rr = []
    for i in range(sim.size(0)):
        rank_i = (ranks[i] == i).nonzero(as_tuple=False).item() + 1
        rr.append(rank_i)
    out["MedR"] = float(np.median(rr))
    return out

def ndcg_at_k(sim, k=10):
    ranks = sim.argsort(dim=1, descending=True)
    dcgs = []
    for i in range(sim.size(0)):
        # relevance 1 at correct index, 0 otherwise
        idx = (ranks[i] == i).nonzero(as_tuple=False).item()
        if idx < k:
            dcg = 1.0 / np.log2(idx + 2)
        else:
            dcg = 0.0
        dcgs.append(dcg)
    return {"nDCG@10": float(np.mean(dcgs))}

def main(cfg_path: str, ckpt_path: str):
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
    sim = zi_val @ zt_val.t()
    it = recall_at_k(sim); it.update(ndcg_at_k(sim))

    ti = recall_at_k(sim.t()); ti.update(ndcg_at_k(sim.t()))

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
