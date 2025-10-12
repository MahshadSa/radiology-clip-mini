from __future__ import annotations
import argparse
from pathlib import Path
from time import time

import torch
import torch.optim as optim
import yaml
from torch.amp import GradScaler
from .data import build_dataloaders
from .models import CLIPMini, clip_loss
from .utils import set_seed, git_hash, make_run_dir, write_latest, save_json


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def _val_r1(model: CLIPMini, dls, device: str) -> float:
    model.eval()
    zs_img, zs_txt = [], []
    for b in dls["val"]:
        imgs = b["images"].to(device, non_blocking=True)
        toks = model.text_enc.tokenize(b["texts"]).to(device)
        _, zi, zt = model(imgs, toks)
        zs_img.append(zi.cpu())
        zs_txt.append(zt.cpu())
    if not zs_img:
        return 0.0
    zi = torch.cat(zs_img, 0)
    zt = torch.cat(zs_txt, 0)
    sim = zi @ zt.t()
    ranks = sim.argsort(dim=1, descending=True)
    return float((ranks[:, 0] == torch.arange(sim.size(0))).float().mean().item())


def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg["seed"]))

    dls = build_dataloaders(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPMini(
        embed_dim=cfg["model"]["embed_dim"],
        text_model=cfg["model"]["text_encoder"],
        tau_init=cfg["model"]["temperature_init"],
    ).to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler(device_type=device, enabled=(device == "cuda" and bool(cfg["train"].get("amp", True))))
    pin = (device == "cuda")
    run_dir = make_run_dir(cfg["paths"]["results_dir"])
    save_json(
        {
            "git": git_hash(),
            "cfg_path": cfg_path,
            "cfg": cfg,
            "start_time": int(time()),
            "device": device,
        },
        Path(run_dir) / "run.json",
    )

    best_r1 = -1.0
    model.train()
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        tot_loss, n = 0.0, 0
        for batch in dls["train"]:
            imgs = batch["images"].to(device, non_blocking=True)
            toks = model.text_enc.tokenize(batch["texts"]).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                sim, _, _ = model(imgs, toks)
                loss = clip_loss(sim)

            scaler.scale(loss).backward()
            if float(cfg["train"]["grad_clip"]) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(opt)
            scaler.update()

            tot_loss += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)

        avg_loss = tot_loss / max(1, n)
        try:
            r1 = _val_r1(model, dls, device)
        except Exception:
            r1 = 0.0

        print(f"epoch {epoch} | loss {avg_loss:.4f} | val R@1 {r1:.3f} | τ {model.temperature.item():.4f}")

        ckpt = Path(run_dir) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        write_latest(ckpt, cfg["paths"]["results_dir"])

        if r1 > best_r1:
            best_r1 = r1
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_R1": r1},
                Path(run_dir) / "best.pt",
            )

    print(f"done. latest → {Path(run_dir) / 'checkpoint.pt'} ; best → {Path(run_dir) / 'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train CLIPMini")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()
    main(args.config)
