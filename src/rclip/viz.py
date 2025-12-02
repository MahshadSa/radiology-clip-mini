from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
import yaml

from .data import build_dataloaders
from .models import CLIPMini


@torch.no_grad()
def retrieval_grid(model: CLIPMini, loader, device: str, out_path: Path, k: int = 5) -> None:
    """Save a grid: query image + top-k retrieved images."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise RuntimeError("Validation loader is empty, cannot build retrieval grid")

    imgs = batch["images"].to(device)
    texts = batch["texts"]
    toks = model.text_enc.tokenize(texts).to(device)

    sim, _, _ = model(imgs, toks)  # (B, B)
    topk = sim.argsort(dim=1, descending=True)[:, :k]

    q = 0
    idxs = topk[q].tolist()
    sel_idx = [q] + idxs
    sel = imgs[sel_idx]

    grid = vutils.make_grid(sel, nrow=k + 1, pad_value=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)


@torch.no_grad()
def gradcam_last_block(model: CLIPMini, img: torch.Tensor, device: str) -> np.ndarray:
    """Return heatmap (H, W) for last conv block, resized to image size."""
    feats = {}

    def hook(_, __, output):
        feats["act"] = output.detach()

    handle = model.image_enc.backbone[-1].register_forward_hook(hook)
    model.eval()

    img_b = img.unsqueeze(0).to(device)  # (1, C, H, W)
    tokens = model.text_enc.tokenize(["dummy"]).to(device)
    _ = model(img_b, tokens)
    handle.remove()

    act = feats["act"]  # (1, C, h, w)
    heat = act.mean(dim=1, keepdim=True)  # (1, 1, h, w)

    H, W = img.shape[1], img.shape[2]
    heat = torch.nn.functional.interpolate(
        heat,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)

    heat = heat.squeeze(0).squeeze(0)  # (H, W)

    heat = heat.cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat


def _unnormalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)[:, None, None]
    return img * std + mean


def overlay_heatmap(img: torch.Tensor, heat: np.ndarray) -> Image.Image:
    """Blend unnormalized image with heatmap. Both must be (H, W)."""
    img = _unnormalize_imagenet(img)
    base = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)

    heat_rgb = (np.stack([heat] * 3, axis=-1) * 255).astype(np.uint8)  # (H, W, 3)

    alpha = 0.4
    out = (alpha * heat_rgb + (1.0 - alpha) * base).astype(np.uint8)
    return Image.fromarray(out)


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

    out_dir = Path(ckpt_path).parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # retrieval grid
    grid_path = out_dir / "retrieval_grid.png"
    retrieval_grid(model, dls["val"], device, grid_path, k=5)

    # GradCAM example
    batch = next(iter(dls["val"]))
    img = batch["images"][0]
    heat = gradcam_last_block(model, img, device)
    overlay = overlay_heatmap(img, heat)
    overlay_path = out_dir / "gradcam_example.png"
    overlay.save(overlay_path)

    print(f"saved {grid_path} and {overlay_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
