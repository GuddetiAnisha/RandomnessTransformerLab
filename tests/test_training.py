import numpy as np
import torch
from torch.utils.data import DataLoader

from randomness_lab.data import BinarySequenceDataset
from randomness_lab.model import CompactBitTransformer
from randomness_lab.training import evaluate_model, train_model


def test_small_training_pipeline_runs():
    bits = np.resize(np.array([0, 0, 1, 1], dtype=np.uint8), 400)
    dataset = BinarySequenceDataset(bits, context_length=8, target_bits=1, stride=2)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = CompactBitTransformer(8, 1, d_model=8, nhead=2, num_layers=1,
                                  dim_feedforward=16, dropout=0)
    result = train_model(model, loader, loader, epochs=1, patience=1, device="cpu")
    metrics = evaluate_model(model, loader, device="cpu")
    assert len(result["history"]) == 1
    assert 0 <= metrics["accuracy"] <= 1
    assert np.isfinite(metrics["cross_entropy_bits_per_bit"])
