from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

from .data import build_dataloaders
from .models import CLIPMini
from .utils import save_json

@torch.no_grad()
def retrieval_grid(model, loader, device, out_path: Path, k=5):
    batch = next(iter(loader))
    imgs = batch["images"].to(device)
    texts= batch["texts"]
    toks = model.text_enc.tokenize(texts).to(device)

    sim, zi, zt = model(imgs, toks)  # (B, B)
    topk = sim.argsort(dim=1, descending=True)[:, :k]
    q = 0
    idxs = topk[q].tolist()
    sel = imgs[[q] + idxs]  # (k+1, C, H, W)
    grid = vutils.make_grid(sel, nrow=k+1, pad_value=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)

def gradcam_last_block(model: CLIPMini, img: torch.Tensor, device):
    """Return heatmap (H,W) for last conv block."""
    feats = {}
    def hook(m, i, o): feats["act"] = o.detach()
    handle = model.image_enc.backbone[-1].register_forward_hook(hook)
    model.eval()
    img = img.unsqueeze(0).to(device)
    tokens = model.text_enc.tokenize(["dummy"]).to(device)
    sim, _, _ = model(img, tokens)  
    handle.remove()
    act = feats["act"].squeeze(0)       # (C, h, w)
    heat = act.mean(dim=0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat

def overlay_heatmap(img: torch.Tensor, heat: np.ndarray):
    """Return PIL image overlay."""
    base = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    heat_rgb = (np.stack([heat]*3, axis=-1)*255).astype(np.uint8)
    alpha = 0.4
    out = (alpha*heat_rgb + (1-alpha)*base).astype(np.uint8)
    return Image.fromarray(out)

def main(cfg_path: str, ckpt_path: str):
    import yaml
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
    model.load_state_dict(state["model"]); model.eval()

    out_dir = Path(ckpt_path).parent / "viz"
    retrieval_grid(model, dls["val"], device, out_dir / "retrieval_grid.png", k=5)

    # one Grad-CAM example (first image of first val batch)
    img = next(iter(dls["val"]))["images"][0]
    heat = gradcam_last_block(model, img, device)
    overlay = overlay_heatmap(img, heat)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(out_dir / "gradcam_example.png")
    print(f"saved {out_dir/'retrieval_grid.png'} and {out_dir/'gradcam_example.png'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
