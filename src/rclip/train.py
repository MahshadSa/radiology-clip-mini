from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from src.data import load_iu_xray_flat, save_splits_json
from src.models import CLIPMiniModel, ModelConfig, contrastive_loss

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_image_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),  
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class PairDataset(Dataset):
    """
    Tiny dataset that holds (image, text) lists and a list of indices (split).
    """
    def __init__(self, images: List, texts: List[str], indices: List[int], transform: T.Compose):
        self.images = images
        self.texts = texts
        self.idx = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        img = self.transform(self.images[j])
        txt = self.texts[j]
        return img, txt, j  # j = absolute index (for debugging if needed)


def make_collate_fn(tokenizer: DistilBertTokenizerFast, max_length: int = 128):
    def collate(batch):
        imgs, txts, ids = zip(*batch)             
        imgs = torch.stack(imgs, dim=0)            # (B, 3, H, W)
        tok = tokenizer(
            list(txts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return imgs, tok["input_ids"], tok["attention_mask"]
    return collate


# Metrics
@torch.no_grad()
def recall_at_1(
    model: CLIPMiniModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute Recall@1 over the whole split.
    """
    model.eval()
    img_embs, txt_embs = [], []
    for images, input_ids, attention_mask in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img_e, txt_e, _ = model(images, input_ids, attention_mask, compute_logits=False)
        img_embs.append(img_e.cpu())
        txt_embs.append(txt_e.cpu())
    img = torch.cat(img_embs, dim=0) 
    txt = torch.cat(txt_embs, dim=0) 
    sims = img @ txt.t()               # cosine sims because L2-normalized
    preds = sims.argmax(dim=1)         
    labels = torch.arange(sims.size(0))
    r1 = (preds == labels).float().mean().item()
    return r1



def train_one_epoch(
    model: CLIPMiniModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for images, input_ids, attention_mask in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img_emb, txt_emb, _ = model(images, input_ids, attention_mask, compute_logits=False)
            loss = contrastive_loss(img_emb, txt_emb, temperature=model.temperature())
        scaler.scale(loss).step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def save_checkpoint(model: CLIPMiniModel, path: Path, extra: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "extra": extra,
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mini CLIP-style model on IU X-Ray (subsample).")
    
    p.add_argument("--splits", type=str, default="data/splits.json", help="Path to JSON with train/val/test indices.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_length", type=int, default=128)
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr_text", type=float, default=3e-5, help="LR for text backbone + text head.")
    p.add_argument("--lr_img_head", type=float, default=1e-3, help="LR for image projection head.")
    p.add_argument("--lr_img_backbone", type=float, default=1e-5, help="LR for image backbone if unfrozen.")
    p.add_argument("--freeze_image_backbone", action="store_true", help="Freeze ResNet-18 backbone.")
    p.add_argument("--freeze_text_backbone", action="store_true", help="Freeze DistilBERT backbone.")
    
    p.add_argument("--results_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_iu_xray_flat()  
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    tfm = build_image_transform(size=args.image_size)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_ds = PairDataset(data["image"], data["text"], splits["train"], tfm)
    val_ds   = PairDataset(data["image"], data["text"], splits["val"], tfm)

    collate = make_collate_fn(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, collate_fn=collate)

    cfg = ModelConfig(
        text_encoder_name="distilbert-base-uncased",
        embed_dim=256,
        freeze_image_backbone=args.freeze_image_backbone,
        freeze_text_backbone=args.freeze_text_backbone,
        learnable_temperature=True,
        init_temperature=0.07,
    )
    model = CLIPMiniModel(cfg).to(device)


    params = []
    # text backbone + text head
    text_params = list(model.text_enc.parameters()) + list(model.txt_proj.parameters())
    params.append({"params": (p for p in text_params if p.requires_grad), "lr": args.lr_text})
    # image projection head
    params.append({"params": (p for p in model.img_proj.parameters() if p.requires_grad), "lr": args.lr_img_head})
    # image backbone (if not frozen)
    img_backbone_params = [p for p in model.image_enc.parameters() if p.requires_grad]
    if img_backbone_params:
        params.append({"params": img_backbone_params, "lr": args.lr_img_backbone})
    # temperature
    if getattr(model, "log_temp", None) is not None:
        params.append({"params": [model.log_temp], "lr": args.lr_img_head})

    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)


    total_steps = math.ceil(len(train_loader) * args.epochs)
    warmup_steps = max(10, total_steps // 20)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())



    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_r1 = -1.0
    metrics = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()

        r1 = recall_at_1(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_R@1={r1:.3f}  tau={model.temperature().item():.3f}")
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_R1": r1, "temperature": model.temperature().item()})

        # Save best
        if r1 > best_r1:
            best_r1 = r1
            save_checkpoint(model, run_dir / "checkpoint.pt", extra={
                "epoch": epoch,
                "config": asdict(cfg),
                "val_R1": r1,
                "args": vars(args),
            })

        
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(f"Best val R@1: {best_r1:.3f}")
    print(f"Run artifacts in: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
