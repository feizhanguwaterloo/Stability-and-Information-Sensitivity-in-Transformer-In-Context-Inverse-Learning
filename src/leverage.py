"""Sufficient-statistics estimators and matched interventions for Experiment G."""

import numpy as np
import torch
from scipy.special import logsumexp
from torch import nn

from .data import HIGH, LOW
from .dispatch import INTERIOR, solve_dispatch
from .kkt import equality_gram


def sufficient_statistics(obs):
    x = obs.dispatch
    weighted_price = obs.price[:, None] * (obs.status == INTERIOR)
    rhs = np.stack((weighted_price.mean(axis=0), (x * weighted_price).mean(axis=0)), axis=1)
    return equality_gram(obs), rhs


def ridge_estimate(obs, prior, penalty):
    gram, rhs = sufficient_statistics(obs)
    return np.linalg.solve(gram + penalty * np.eye(2),
                           (rhs + penalty * prior.reshape(2, 2))[..., None]).ravel()


class StatisticsMLP(nn.Module):
    """A small MLP with the interior normal equations as its input."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(10, 96), nn.ReLU(), nn.Linear(96, 96),
                                  nn.ReLU(), nn.Linear(96, 4))
        self.register_buffer("low", torch.tensor(LOW, dtype=torch.float32))
        self.register_buffer("span", torch.tensor(HIGH - LOW, dtype=torch.float32))

    def forward(self, records):
        x, price = records[..., 1:3], records[..., 3:4]
        interior = records[..., [5, 8]]
        # The price features are scaled by PRICE_SCALE, an invertible fixed scaling.
        stats = torch.stack((interior, interior * x, interior * x.square(),
                             interior * price, interior * price * x), dim=-1)
        pooled = stats.mean(dim=1).flatten(start_dim=1)
        return self.low + self.span * torch.sigmoid(self.head(pooled))


def theorem_pair(m):
    if m < 2:
        raise ValueError("at least two records are required")
    theta = np.array([[1.6, 1.0, 1.6, 1.0], [1.2, 1.8, 1.2, 1.8]])
    demand = np.r_[np.ones(m - 1), 1.6]
    return theta, [solve_dispatch(task, demand) for task in theta]


def crossed_backgrounds(theta, m):
    """Cross the same focal record with symmetric near and spread backgrounds."""
    grid = solve_dispatch(theta, np.linspace(0, 2, 2049)[1:-1])
    candidates = np.flatnonzero(np.all(grid.status == INTERIOR, axis=1))
    center = (candidates[0] + candidates[-1]) // 2
    radius = min(center - candidates[0], candidates[-1] - center)
    narrow = np.arange(-(m // 2 - 1), m // 2)
    spread = np.rint(np.linspace(-.85 * radius, .85 * radius, m - 1)).astype(int)
    for side in (-1, 1):
        start = center + side * round(.60 * radius)
        stop = center + side * round(.73 * radius)
        for label, background in (("high-leverage", narrow), ("low-leverage", spread)):
            yield side, label, grid.take(np.r_[center + background, start]), grid.take([stop])


def hat_leverage(obs):
    """Maximum focal-record leverage over the two interior regressions."""
    gram = equality_gram(obs)
    vector = np.column_stack((np.ones(2), obs.dispatch[-1]))
    values = np.einsum("gi,gij,gj->g", vector, np.linalg.pinv(gram, rcond=1e-12), vector)
    return np.max(values) / len(obs.demand)


def condition_number(obs):
    eigenvalues = np.linalg.eigvalsh(equality_gram(obs))
    smallest = eigenvalues[:, 0]
    if np.any(smallest <= 1e-12):
        return np.inf
    return np.max(eigenvalues[:, 1] / smallest)


def norm_upper(weight):
    """Two algebraic upper bounds on the spectral norm."""
    weight = np.asarray(weight.detach().cpu(), dtype=np.float64)
    absolute = np.abs(weight)
    return min(np.linalg.norm(weight), np.sqrt(absolute.sum(axis=0).max() *
                                               absolute.sum(axis=1).max()))


def vector_norm(vector):
    return np.linalg.norm(np.asarray(vector.detach().cpu(), dtype=np.float64))


def transformer_bound(model):
    """Return log10(C) for the real-arithmetic weight-norm bound C/m."""
    embedding = norm_upper(model.embedding.weight)
    radius = np.sqrt(6) * embedding + vector_norm(model.embedding.bias)
    log_constant = np.log(2 * np.sqrt(6) * embedding)
    max_score = 0.0
    for layer in model.encoder.layers:
        attn = layer.self_attn
        head_size = attn.embed_dim // attn.num_heads
        qkv = attn.in_proj_weight.chunk(3)
        biases = attn.in_proj_bias.chunk(3)
        head_logs = []
        for head in range(attn.num_heads):
            sl = slice(head * head_size, (head + 1) * head_size)
            q, k, v = (norm_upper(weight[sl]) for weight in qkv)
            qr, kr, vr = (operator * radius + vector_norm(bias[sl])
                          for operator, bias in zip((q, k, v), biases))
            scale = np.sqrt(head_size)
            score = qr * kr / scale
            max_score = max(max_score, score)
            coefficient = v + 2 * vr * (q * kr + k * qr) / scale
            head_logs.append(2 * score + np.log(coefficient))
        log_attention = np.log(norm_upper(attn.out_proj.weight)) + logsumexp(head_logs)
        gain1 = float(layer.norm1.weight.detach().abs().max())
        gain2 = float(layer.norm2.weight.detach().abs().max())
        log_constant += np.logaddexp(0, log_attention)
        log_constant += np.log(gain1 / np.sqrt(layer.norm1.eps))
        log_constant += np.log1p(norm_upper(layer.linear1.weight) *
                                 norm_upper(layer.linear2.weight))
        log_constant += np.log(gain2 / np.sqrt(layer.norm2.eps))
        radius = np.sqrt(attn.embed_dim) * gain2 + vector_norm(layer.norm2.bias)
    log_constant += np.log(float(model.span.max()) * norm_upper(model.head.weight) / 4)
    return log_constant / np.log(10), max_score
