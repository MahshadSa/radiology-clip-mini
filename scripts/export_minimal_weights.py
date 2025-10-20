#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml

# local imports (src layout)
from pgr.models import CLIPMini  


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint(ckpt_path: str, device: str = "cpu"):
    state = torch.load(ckpt_path, map_location=device)
    return state["model"] if isinstance(state, dict) and "model" in state else state


def export_full(ckpt_path: str, out_path: str):
    sd = _load_checkpoint(ckpt_path)
    torch.save({"model": sd}, out_path)
    print(f"[full] wrote: {out_path}")


def export_minimal(cfg: dict, ckpt_path: str, out_path: str, device: str = "cpu"):
    """
    Minimal export: store only small heads + config to rebuild base encoders.
    """
    model = CLIPMini(
        embed_dim=cfg["model"]["embed_dim"],
        text_model=cfg["model"]["text_encoder"],
        tau_init=cfg["model"]["temperature_init"],
    ).to(device)

    sd = _load_checkpoint(ckpt_path, device)
    model.load_state_dict(sd, strict=False)

    payload = {
        "format": "rclip-mini/1",
        "text_model": cfg["model"]["text_encoder"],
        "embed_dim": int(cfg["model"]["embed_dim"]),
        "image_proj": model.image_enc.proj.state_dict(),
        "text_proj": model.text_enc.proj.state_dict(),
        "log_tau": model.log_tau.detach().cpu(),
    }
    torch.save(payload, out_path)
    print(f"[minimal] wrote: {out_path}")
    print("Contents:", {k: type(v).__name__ for k, v in payload.items()})


def main():
    ap = argparse.ArgumentParser(description="Export CLIPMini weights")
    ap.add_argument("--config", required=True, help="YAML config used to build the model")
    ap.add_argument("--ckpt", required=True, help="Path to training checkpoint (best.pt/checkpoint.pt)")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--mode", choices=["full", "minimal"], default="minimal",
                    help="Export full state_dict or minimal heads+config")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(args.config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        export_full(args.ckpt, args.out)
    else:
        export_minimal(cfg, args.ckpt, args.out, device=device)


if __name__ == "__main__":
    main()
