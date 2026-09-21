from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def block_to_class(block: np.ndarray) -> int:
    value = 0
    for bit in block:
        value = (value << 1) | int(bit)
    return value


class BinarySequenceDataset(Dataset):
    def __init__(self, bits: np.ndarray, context_length: int = 64,
                 target_bits: int = 1, stride: int = 1):
        self.bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        self.context_length = context_length
        self.target_bits = target_bits
        self.starts = np.arange(0, len(self.bits) - context_length - target_bits + 1, stride)
        if len(self.starts) == 0:
            raise ValueError("Sequence is too short for context and target sizes")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        context = torch.from_numpy(self.bits[start:start + self.context_length].astype(np.int64))
        block = self.bits[start + self.context_length:start + self.context_length + self.target_bits]
        target = torch.tensor(block_to_class(block), dtype=torch.long)
        return context, target


def contiguous_split(bits: np.ndarray, train_fraction: float = 0.70,
                     validation_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split")
    first = int(len(bits) * train_fraction)
    second = int(len(bits) * (train_fraction + validation_fraction))
    return bits[:first], bits[first:second], bits[second:]
