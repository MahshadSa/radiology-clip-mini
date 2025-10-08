from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import DistilBertTokenizerFast

from src.data import load_iu_xray_flat
from src.models import CLIPMiniModel, ModelConfig


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
    def __init__(self, images: List, texts: List[str], indices: List[int], transform: T.Compose):
        self.images = images
        self.texts = texts
        self.idx = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        return self.transform(self.images[j]), self.texts[j], j


def make_collate_fn(tokenizer: DistilBertTokenizerFast, max_length: int = 128):
    def collate(batch):
        imgs, txts, ids = zip(*batch)
        imgs = torch.stack(imgs, 0)
        tok = tokenizer(
            list(txts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return imgs, tok["input_ids"], tok["attention_mask"]
    return collate



@torch.no_grad()
def embed_split(
    model: CLIPMiniModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (image_embeddings, text_embeddings) for the entire split.
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
    return torch.cat(img_embs, 0), torch.cat(txt_embs, 0)  # (N, D), (N, D)


def _recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return (ranks < k).float().mean().item()

def _median_rank(ranks: torch.Tensor) -> float:
    return ranks.median().item()

def _ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    """
    nDCG@k for single-relevance (1 if the true pair is within top-k, else 0),
    with relevance=1 at the true rank. DCG = 1 / log2(1+rank+1).
    If rank >= k → DCG = 0. IDCG = 1.0 (best case rank=0).
    """
    gains = torch.zeros_like(ranks, dtype=torch.float)
    mask = ranks < k
    gains[mask] = 1.0 / torch.log2(ranks[mask].float() + 2.0)  # +2 for (rank+1) and 1-based log
    # IDCG with single relevant item is 1.0 (rank 0 ⇒ 1/log2(1+1)=1)
    return gains.mean().item()


def compute_directional_metrics(sim: torch.Tensor) -> Dict[str, float]:
    """
    sim: (N, N) similarity matrix (higher is better).
    Returns metrics for "query rows → candidate columns".
    True match for row i is column i.
    """
    sorted_idx = torch.argsort(sim, dim=1, descending=True)  
    arange = torch.arange(sim.size(0)).unsqueeze(1)          # (N,1)
    matches = (sorted_idx == arange)                         # True at the column where col==i
    ranks = matches.float().argmax(dim=1)                    # 0-based rank positions

    return {
        "R@1":  _recall_at_k(ranks, 1),
        "R@5":  _recall_at_k(ranks, 5),
        "R@10": _recall_at_k(ranks, 10),
        "MedR": _median_rank(ranks),
        "nDCG@10": _ndcg_at_k(ranks, 10),
    }


# CLI / main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CLIP-style retrieval metrics on IU X-Ray split.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pt from training.")
    p.add_argument("--splits", type=str, default="data/splits.json", help="Path to JSON with train/val/test indices.")
    p.add_argument("--which_split", type=str, default="val", choices=["train", "val", "test"],
                   help="Which split to evaluate.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--results_dir", type=str, default=None,
                   help="Where to save metrics.json. Default: alongside ckpt (run folder).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_iu_xray_flat()
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_indices = splits[args.which_split]

    tfm = build_image_transform(args.image_size)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ds = PairDataset(data["image"], data["text"], split_indices, tfm)
    collate = make_collate_fn(tok, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=collate)

    ckpt_path = Path(args.ckpt)
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = payload.get("extra", {}).get("config", None)
    cfg = ModelConfig(**cfg_dict) if cfg_dict is not None else ModelConfig()
    model = CLIPMiniModel(cfg).to(device)
    model.load_state_dict(payload["model_state"], strict=True)

    img_emb, txt_emb = embed_split(model, loader, device)

    # Similarities (cosine because embeddings are L2-normalized)
    sim = img_emb @ txt_emb.t()  # (N, N)

    metrics_i2t = compute_directional_metrics(sim)       # image → text
    metrics_t2i = compute_directional_metrics(sim.t())   # text  → image

    metrics = {
        "split": args.which_split,
        "num_pairs": int(sim.size(0)),
        "image_to_text": metrics_i2t,
        "text_to_image": metrics_t2i,
    }

    if args.results_dir:
        out_dir = Path(args.results_dir)
    else:
        out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    def fmt(m: Dict[str, float]) -> str:
        return f'R@1={m["R@1"]:.3f}  R@5={m["R@5"]:.3f}  R@10={m["R@10"]:.3f}  MedR={m["MedR"]:.1f}  nDCG@10={m["nDCG@10"]:.3f}'
    print(f"[{args.which_split}]  I→T  {fmt(metrics_i2t)}")
    print(f"[{args.which_split}]  T→I  {fmt(metrics_t2i)}")
    print(f"Saved metrics to: {(out_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
