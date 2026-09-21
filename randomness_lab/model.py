from __future__ import annotations

import torch
from torch import nn


class CompactBitTransformer(nn.Module):
    def __init__(self, context_length: int = 64, target_bits: int = 1,
                 d_model: int = 48, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 1 <= target_bits <= 8:
            raise ValueError("target_bits must be between 1 and 8")
        self.context_length = context_length
        self.target_bits = target_bits
        self.token_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2 ** target_bits)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[1] != self.context_length:
            raise ValueError(f"Expected context length {self.context_length}")
        values = self.token_embedding(bits) + self.position_embedding
        mask = torch.triu(torch.ones(self.context_length, self.context_length,
                                     device=bits.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(values, mask=mask)
        return self.classifier(self.norm(encoded[:, -1]))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
