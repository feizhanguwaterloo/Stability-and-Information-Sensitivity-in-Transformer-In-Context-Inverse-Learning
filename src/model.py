from math import isfinite

import torch
from torch import nn

from .data import HIGH, LOW


class SetTransformer(nn.Module):
    """Estimate boxed cost parameters from an unordered market context."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(10, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(64, 4)
        self.register_buffer("low", torch.tensor(LOW, dtype=torch.float32))
        self.register_buffer("span", torch.tensor(HIGH - LOW, dtype=torch.float32))

    def forward(self, records):
        states = self.encoder(self.embedding(records))
        unit_box = torch.sigmoid(self.head(states.mean(dim=1)))
        return self.low + self.span * unit_box


def attention_operators(model):
    for layer in model.encoder.layers:
        attention = layer.self_attn
        yield from attention.in_proj_weight.chunk(3, dim=0)
        yield attention.out_proj.weight


class NormControlledTransformer(SetTransformer):
    def __init__(self, cap):
        super().__init__()
        if not isfinite(cap) or cap <= 0:
            raise ValueError("attention norm cap must be finite and positive")
        self.cap = cap

    @torch.no_grad()
    def project_attention(self):
        for weight in attention_operators(self):
            scale = (torch.linalg.matrix_norm(weight, ord=2) / self.cap).clamp_min(1)
            weight.div_(scale)


class DeepSets(nn.Module):
    """Parameter-matched rowwise MLP followed by mean pooling."""

    def __init__(self):
        super().__init__()
        self.rows = nn.Sequential(nn.Linear(10, 216), nn.ReLU(), nn.Linear(216, 216),
                                  nn.ReLU(), nn.Linear(216, 64), nn.ReLU())
        self.head = nn.Linear(64, 4)
        self.register_buffer("low", torch.tensor(LOW, dtype=torch.float32))
        self.register_buffer("span", torch.tensor(HIGH - LOW, dtype=torch.float32))

    def forward(self, records):
        pooled = self.rows(records).mean(dim=1)
        return self.low + self.span * torch.sigmoid(self.head(pooled))
