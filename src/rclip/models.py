from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from transformers import AutoModel



def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class ModelConfig:
    text_encoder_name: str = "distilbert-base-uncased"
    embed_dim: int = 256
    freeze_image_backbone: bool = False
    freeze_text_backbone: bool = False
    learnable_temperature: bool = True
    init_temperature: float = 0.07  


# Encoders + projection heads
class ImageEncoderResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # expose penultimate features
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class TextEncoderDistilBERT(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.out_dim = self.transformer.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, H)
        return cls


class ProjectionHead(nn.Module):
    """
    Linear projection to the shared embedding space.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CLIPMiniModel(nn.Module):
    """
    Minimal CLIP-style model for image↔text retrieval (ResNet18 + DistilBERT).

    Forward returns:
      image_emb: (B, D) L2-normalized
      text_emb:  (B, D) L2-normalized
      logits:    (B, B) similarity matrix (optional when compute_logits=True)
    """
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.image_enc = ImageEncoderResNet18()
        self.text_enc = TextEncoderDistilBERT(self.cfg.text_encoder_name)

        self.img_proj = ProjectionHead(self.image_enc.out_dim, self.cfg.embed_dim)
        self.txt_proj = ProjectionHead(self.text_enc.out_dim, self.cfg.embed_dim)

        if self.cfg.learnable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(self.cfg.init_temperature)))
            self.register_buffer("fixed_temp", torch.tensor(self.cfg.init_temperature), persistent=False)
        else:
            self.register_buffer("fixed_temp", torch.tensor(self.cfg.init_temperature))
            self.log_temp = None

        if self.cfg.freeze_image_backbone:
            for p in self.image_enc.parameters():
                p.requires_grad = False
        if self.cfg.freeze_text_backbone:
            for p in self.text_enc.parameters():
                p.requires_grad = False

    def temperature(self) -> torch.Tensor:
        if self.log_temp is not None:
            # Clamp for stability
            return torch.clamp(self.log_temp.exp(), 1e-3, 1.0)
        return self.fixed_temp

    # Encoders → embeddings
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_enc(images)     
        emb = self.img_proj(feats)         # (B, D)
        return l2_normalize(emb)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = self.text_enc(input_ids, attention_mask)  
        emb = self.txt_proj(feats)                        # (B, D)
        return l2_normalize(emb)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        compute_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)
        if not compute_logits:
            return img_emb, txt_emb, None
        temp = self.temperature()
        logits = (img_emb @ txt_emb.t()) / temp
        return img_emb, txt_emb, logits


def contrastive_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style):
      L = (CE(sim(I,T)/τ, labels) + CE(sim(T,I)/τ, labels)) / 2
    """
    logits = (image_emb @ text_emb.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


def clip_mini_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Same loss when logits already include the 1/τ scaling."""
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)
