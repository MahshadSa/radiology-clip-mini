# src/viz.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
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

def build_image_transform_for_overlay(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize(size),
        T.CenterCrop(size),
    ])



class PairDatasetViz(Dataset):
    """
    Returns:
      - normalized tensor for model input
      - clean text (string)
      - RGB PIL image (resized/cropped) for plotting/overlay
      - absolute index
    """
    def __init__(self, images: List[Image.Image], texts: List[str], indices: List[int],
                 model_transform: T.Compose, overlay_transform: T.Compose):
        self.images = images
        self.texts = texts
        self.idx = list(indices)
        self.tf_model = model_transform
        self.tf_overlay = overlay_transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        im_pil_overlay = self.tf_overlay(self.images[j])    # PIL RGB (H,W,3)
        im_tensor = self.tf_model(self.images[j])           # (3,H,W) normalized
        txt = self.texts[j]
        return im_tensor, txt, im_pil_overlay, j


def make_collate_fn(tokenizer: DistilBertTokenizerFast, max_length: int = 128):
    def collate(batch):
        imgs_t, txts, imgs_pil, ids = zip(*batch)
        imgs_t = torch.stack(imgs_t, 0)
        tok = tokenizer(
            list(txts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return imgs_t, tok["input_ids"], tok["attention_mask"], list(imgs_pil), ids
    return collate



@torch.no_grad()
def embed_split(
    model: CLIPMiniModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Image.Image]]:
    """
    Return (image_embeddings, text_embeddings, pil_images_in_loader_order).
    Embeddings are L2-normalized already.
    """
    model.eval()
    img_embs, txt_embs, pil_accum = [], [], []
    for images, input_ids, attention_mask, pil_list, _ in loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img_e, txt_e, _ = model(images, input_ids, attention_mask, compute_logits=False)
        img_embs.append(img_e.cpu())
        txt_embs.append(txt_e.cpu())
        pil_accum.extend(pil_list)
    return torch.cat(img_embs, 0), torch.cat(txt_embs, 0), pil_accum  # (N,D), (N,D), [PIL]*N



def draw_image_to_text_grid(
    query_idx: int,
    pil_image: Image.Image,
    topk_texts: List[str],
    sims: List[float],
    out_path: Path,
):
    """
    Left: query image; Right: top-k texts (with sim scores).
    """
    k = len(topk_texts)
    fig = plt.figure(figsize=(10, 2 + 1.2 * k))

    ax_img = plt.subplot2grid((k, 3), (0, 0), rowspan=k, colspan=1)
    ax_img.imshow(pil_image)
    ax_img.axis("off")
    ax_img.set_title(f"Query Image #{query_idx}")

    # Right texts
    ax_txt = plt.subplot2grid((k, 3), (0, 1), rowspan=k, colspan=2)
    ax_txt.axis("off")
    lines = []
    for i, (t, s) in enumerate(zip(topk_texts, sims), start=1):
        t_short = (t[:180] + "…") if len(t) > 180 else t
        lines.append(f"{i}. ({s:.2f}) {t_short}")
    txt = "\n\n".join(lines)
    ax_txt.text(0.02, 0.98, txt, va="top", ha="left", wrap=True, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_text_to_image_grid(
    query_idx: int,
    query_text: str,
    topk_images: List[Image.Image],
    sims: List[float],
    out_path: Path,
):
    """
    Top: query text; Bottom: row of top-k images with similarity scores.
    """
    k = len(topk_images)
    fig = plt.figure(figsize=(2.5 * k, 4.5))
    ax_txt = plt.subplot2grid((2, 1), (0, 0))
    ax_txt.axis("off")
    t_short = (query_text[:260] + "…") if len(query_text) > 260 else query_text
    ax_txt.set_title(f"Query Text #{query_idx}", pad=6)
    ax_txt.text(0.01, 0.8, t_short, va="top", ha="left", wrap=True, fontsize=10)

    # Images row
    for i in range(k):
        ax = plt.subplot2grid((2, k), (1, i))
        ax.imshow(topk_images[i])
        ax.axis("off")
        ax.set_title(f"{i+1} ({sims[i]:.2f})", fontsize=9, pad=3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_retrieval_grids(
    sim: torch.Tensor,
    pil_images: List[Image.Image],
    texts: List[str],
    out_dir: Path,
    top_k: int = 5,
    num_queries: int = 5,
):
    """
    Create image→text and text→image grids for the first `num_queries` items.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    N = sim.size(0)
    q = min(num_queries, N, 10)

    # Image → Text
    i2t_idx = torch.argsort(sim, dim=1, descending=True)  # (N,N)
    for qi in range(q):
        top_idx = i2t_idx[qi, :top_k].tolist()
        top_texts = [texts[j] for j in top_idx]
        top_sims = sim[qi, top_idx].tolist()
        draw_image_to_text_grid(qi, pil_images[qi], top_texts, top_sims,
                                out_dir / f"i2t_query{qi}_top{top_k}.png")

    # Text → Image
    t2i_idx = torch.argsort(sim.t(), dim=1, descending=True)  # (N,N)
    for qi in range(q):
        top_idx = t2i_idx[qi, :top_k].tolist()
        top_imgs = [pil_images[j] for j in top_idx]
        top_sims = sim[top_idx, qi].tolist()
        draw_text_to_image_grid(qi, texts[qi], top_imgs, top_sims,
                                out_dir / f"t2i_query{qi}_top{top_k}.png")


class ResNet18GradCAM:
    """
    Grad-CAM on ResNet-18 last conv layer (layer4[-1].conv2).
    Given an image tensor and a target text embedding,
    we backprop the similarity score to get a heatmap.
    """
    def __init__(self, model: CLIPMiniModel, layer_name: str = "image_enc.backbone.layer4.1.conv2"):
        self.model = model
        self.layer = self._get_layer(layer_name)
        self.activations = None
        self.gradients = None
        self.hook_f = self.layer.register_forward_hook(self._save_activation)
        self.hook_b = self.layer.register_backward_hook(self._save_gradient)

    def _get_layer(self, name: str):
        mod = self.model
        for part in name.split("."):
            mod = getattr(mod, part)
        return mod

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()           # (B, C, H, W)

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()     # (B, C, H, W)

    def remove(self):
        self.hook_f.remove()
        self.hook_b.remove()

    def __call__(self, image_tensor: torch.Tensor, text_embedding: torch.Tensor) -> np.ndarray:
        """
        image_tensor: (1,3,H,W) normalized
        text_embedding: (1,D) L2-normalized (or will be normalized here)
        Returns heatmap (H,W) in [0,1].
        """
        self.model.zero_grad(set_to_none=True)
        img_emb = self.model.encode_image(image_tensor)              # (1,D)
        txt_emb = F.normalize(text_embedding, dim=-1)                # ensure norm 1
        score = (img_emb * txt_emb).sum()                            # scalar similarity
        score.backward()

        A = self.activations[0]   # (C,H,W)
        dA = self.gradients[0]    # (C,H,W)
        weights = dA.mean(dim=(1, 2))                                # (C,)

        cam = F.relu((weights.view(-1, 1, 1) * A).sum(dim=0))        # (H,W)
        cam = cam.cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_heatmap_on_pil(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """
    Overlay a heatmap (H,W in [0,1]) on top of a PIL RGB image. Returns a PIL image.
    """
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]  # (H,W,3) float in [0,1]
    base = np.asarray(pil_img).astype(np.float32) / 255.0
    overlay = (1 - alpha) * base + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_gradcam_for_pair(
    model: CLIPMiniModel,
    image_tensor: torch.Tensor,
    pil_image: Image.Image,
    text_embedding: torch.Tensor,
    out_path: Path,
):
    """
    Compute Grad-CAM for a single (image, text) pair and save overlay PNG.
    """
    device = next(model.parameters()).device
    cam_engine = ResNet18GradCAM(model)
    try:
        heatmap = cam_engine(image_tensor.to(device), text_embedding.to(device))
    finally:
        cam_engine.remove()

    overlay = overlay_heatmap_on_pil(pil_image, heatmap, alpha=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


# CLI / main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualization: retrieval grids and Grad-CAM overlays.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to training checkpoint.pt")
    p.add_argument("--splits", type=str, default="data/splits.json", help="Path to JSON with train/val/test indices.")
    p.add_argument("--which_split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--num_queries", type=int, default=5)
    p.add_argument("--gradcam_for_first", action="store_true",
                   help="Also produce a Grad-CAM for the first query pair in image→text.")
    p.add_argument("--viz_dir", type=str, default=None,
                   help="Output folder for viz. Default: <ckpt_dir>/viz/")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_iu_xray_flat()
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    idxs = splits[args.which_split]

    tf_model = build_image_transform(args.image_size)
    tf_overlay = build_image_transform_for_overlay(args.image_size)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ds = PairDatasetViz(data["image"], data["text"], idxs, tf_model, tf_overlay)
    collate = make_collate_fn(tok, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=collate)

    ckpt_path = Path(args.ckpt)
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = payload.get("extra", {}).get("config", None)
    cfg = ModelConfig(**cfg_dict) if cfg_dict is not None else ModelConfig()
    model = CLIPMiniModel(cfg).to(device)
    model.load_state_dict(payload["model_state"], strict=True)

    img_emb, txt_emb, pil_images = embed_split(model, loader, device)
    sim = img_emb @ txt_emb.t()  

    viz_dir = Path(args.viz_dir) if args.viz_dir else (ckpt_path.parent / "viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    save_retrieval_grids(sim, pil_images, [data["text"][i] for i in idxs],
                         viz_dir, top_k=args.top_k, num_queries=args.num_queries)

    if args.gradcam_for_first and len(idxs) > 0:
        first_im_tensor, _, first_pil, _ = ds[0]
        first_txt_emb = txt_emb[0:1]  # (1,D)
        save_gradcam_for_pair(
            model=model,
            image_tensor=first_im_tensor.unsqueeze(0),  # (1,3,H,W)
            pil_image=first_pil,
            text_embedding=first_txt_emb,
            out_path=viz_dir / "gradcam_first_pair.png",
        )

    print(f"Saved retrieval grids (and Grad-CAM if requested) to: {viz_dir.resolve()}")


if __name__ == "__main__":
    main()
