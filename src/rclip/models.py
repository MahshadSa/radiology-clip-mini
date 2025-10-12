from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # up to last conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        feats = self.backbone(x)                 # (B, 512, H/32, W/32)
        pooled = self.pool(feats).flatten(1)     # (B, 512)
        z = self.proj(pooled)                    # (B, D)
        z = F.normalize(z, dim=-1)
        return z

class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tok = DistilBertTokenizerFast.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def tokenize(self, texts, max_length=128):
        return self.tok(
            texts,
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        )

    def forward(self, batch_tokens):
        out = self.bert(**batch_tokens)          # last_hidden_state 
        cls = out.last_hidden_state[:, 0]        # [CLS]-like token
        z = self.proj(cls)                        # (B, D)
        z = F.normalize(z, dim=-1)
        return z

class CLIPMini(nn.Module):
    def __init__(self, embed_dim=256, text_model="distilbert-base-uncased", tau_init=0.07):
        super().__init__()
        self.image_enc = ImageEncoder(embed_dim)
        self.text_enc  = TextEncoder(embed_dim, text_model)
        # temperature as log-scale parameter (stable learning)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))

    @property
    def temperature(self):
        return self.log_tau.exp().clamp(1e-3, 1.0)

    def forward(self, images, text_tokens):
        zi = self.image_enc(images)              # (B, D)
        zt = self.text_enc(text_tokens)          # (B, D)
        sim = zi @ zt.t()                        # cosine similarity (normalized)
        sim = sim / self.temperature
        return sim, zi, zt

def clip_loss(sim: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over similarity matrix (B, B)."""
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)        # image->text
    loss_t = F.cross_entropy(sim.t(), labels)    # text->image
    return 0.5 * (loss_i + loss_t)
