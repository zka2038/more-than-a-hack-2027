"""Bidirectional GRU over per-frame features: state (H/A/D) logits and a change logit per frame.

Offline decoding is allowed, so the model sees the seconds after a touch: whether the new team
actually gained control (the challenge's possession rule) is visible there.
"""

import torch
from torch import nn


class PossessionNet(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.1, kernel=9):
        super().__init__()
        self.config = dict(n_features=n_features, hidden=hidden, layers=layers, dropout=dropout,
                           kernel=kernel)
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, x):
        """x (B, T, F) -> (state logits (B, T, 3), change logit (B, T))."""
        h = self.inp(x)
        h = self.norm(h + torch.nn.functional.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2))
        o, _ = self.gru(h)
        out = self.head(o)
        return out[..., :3], out[..., 3]
