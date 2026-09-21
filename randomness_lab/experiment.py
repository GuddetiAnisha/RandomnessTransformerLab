from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BinarySequenceDataset, contiguous_split
from .model import CompactBitTransformer
from .statistics import analyze
from .training import evaluate_model, train_model
from .utils import save_json, set_seed


@dataclass
class ExperimentConfig:
    seed: int = 42
    context_length: int = 64
    target_bits: int = 1
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 2
    stride: int = 1


def run_experiment(bits: np.ndarray, config: ExperimentConfig,
                   output_dir: str | Path | None = None) -> dict:
    set_seed(config.seed)
    train_bits, validation_bits, test_bits = contiguous_split(bits)
    datasets = [BinarySequenceDataset(part, config.context_length, config.target_bits, config.stride)
                for part in (train_bits, validation_bits, test_bits)]
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(datasets[0], batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(datasets[1], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=config.batch_size, shuffle=False)
    model = CompactBitTransformer(
        context_length=config.context_length, target_bits=config.target_bits,
        d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward, dropout=config.dropout)
    training = train_model(model, train_loader, validation_loader, config.epochs,
                           config.learning_rate, config.weight_decay, config.patience)
    metrics = evaluate_model(model, test_loader)
    labels = np.array([datasets[2][idx][1].item() for idx in range(len(datasets[2]))])
    counts = np.bincount(labels, minlength=2 ** config.target_bits)
    majority_accuracy = float(counts.max() / counts.sum())
    result = {
        "config": asdict(config),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda_available": torch.cuda.is_available()},
        "sequence_analysis": analyze(bits),
        "split_bits": {"train": len(train_bits), "validation": len(validation_bits), "test": len(test_bits)},
        "model": {"parameters": model.parameter_count, "classes": 2 ** config.target_bits},
        "baseline": {"majority_accuracy": majority_accuracy,
                     "uniform_cross_entropy_bits_per_target": float(config.target_bits)},
        "training": training,
        "test": metrics,
        "improvement_over_majority_accuracy": metrics["accuracy"] - majority_accuracy,
    }
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_json(result, output / "results.json")
        torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, output / "model.pt")
    return result
