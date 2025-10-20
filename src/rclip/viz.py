from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

from .data import build_dataloaders
from .models import CLIPMini


@torch.no_grad()
def retrieval_grid(model: CLIPMini, loader, device: str, out_path: Path, k: int = 5):
    batch = next(iter(loader))
    imgs = batch["images"].to(device, non_blocking=True)
    texts = batch["texts"]
    toks = model.text_enc.tokenize(texts).to(device)

    sim, zi, zt = model(imgs, toks)  # (B, B)
    B = sim.size(0)

    if B == 0:
        return  
    if B == 1:
        sel = imgs 
    else:
        k_eff = max(1, min(k, B - 1))
        topk = sim.argsort(dim=1, descending=True)[:, :k_eff]
        q = 0
        idxs = topk[q].tolist()
        idxs = [i for i in idxs if i != q]
        sel = imgs[[q] + idxs]  # (k_eff+1, C, H, W)

    grid = vutils.make_grid(sel, nrow=sel.size(0), pad_value=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)


@torch.no_grad()
def gradcam_last_block(model: CLIPMini, img: torch.Tensor, device: str) -> np.ndarray:
    """Return an activation-based heatmap (H, W) for the last conv block."""
    feats = {}

    def hook(_m, _i, o):
        feats["act"] = o.detach()

    handle = model.image_enc.backbone[-1].register_forward_hook(hook)
    model.eval()
    img = img.unsqueeze(0).to(device, non_blocking=True)
    tokens = model.text_enc.tokenize(["dummy"]).to(device)
    _ = model(img, tokens)
    handle.remove()

    act = feats["act"].squeeze(0)  # (C, h, w)
    heat = act.mean(dim=0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat


def overlay_heatmap(img: torch.Tensor, heat: np.ndarray) -> Image.Image:
    """Blend a [0,1] CHW float tensor with a heatmap into a PIL image."""
    base = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    heat_rgb = (np.stack([heat] * 3, axis=-1) * 255).astype(np.uint8)
    alpha = 0.4
    out = (alpha * heat_rgb + (1 - alpha) * base).astype(np.uint8)
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
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    out_dir = Path(ckpt_path).parent / "viz"
    retrieval_grid(model, dls["val"], device, out_dir / "retrieval_grid.png", k=5)

    first_batch = next(iter(dls["val"]))
    if len(first_batch["images"]) > 0:
        img = first_batch["images"][0]
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
