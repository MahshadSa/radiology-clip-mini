from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, train_backbone: bool = True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, embed_dim)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        train_bert: bool = True,
    ):
        super().__init__()
        self.tok = DistilBertTokenizerFast.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)
        self.max_length = max_length

        if not train_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, texts) -> dict:
        return self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, batch_tokens: dict) -> torch.Tensor:
        out = self.bert(**batch_tokens)
        cls = out.last_hidden_state[:, 0]
        z = self.proj(cls)
        return F.normalize(z, dim=-1)


class CLIPMini(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        text_model: str = "distilbert-base-uncased",
        tau_init: float = 0.07,
        train_backbone: bool = True,
        train_bert: bool = True,
        max_text_length: int = 128,
    ):
        super().__init__()
        self.image_enc = ImageEncoder(embed_dim, train_backbone=train_backbone)
        self.text_enc = TextEncoder(
            embed_dim=embed_dim,
            model_name=text_model,
            max_length=max_text_length,
            train_bert=train_bert,
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init, dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(1e-3, 1.0)

    def forward(self, images: torch.Tensor, text_tokens: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zi = self.image_enc(images)
        zt = self.text_enc(text_tokens)
        sim = zi @ zt.t()
        sim = sim / self.temperature
        return sim, zi, zt


def clip_loss(sim: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i = F.cross_entropy(sim, labels)
    loss_t = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_i + loss_t)
