from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(targets)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += len(targets)
    return total_loss / total, total_correct / total


def train_model(model, train_loader: DataLoader, validation_loader: DataLoader,
                epochs: int = 5, learning_rate: float = 1e-3,
                weight_decay: float = 1e-4, patience: int = 2,
                device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            validation_loss, validation_accuracy = _run_epoch(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        if validation_loss < best_loss - 1e-5:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "training_seconds": time.perf_counter() - started,
            "best_validation_loss": best_loss, "device": device}


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: str | None = None) -> dict:
    device = device or next(model.parameters()).device
    model.to(device).eval()
    targets_all, probabilities_all = [], []
    latencies = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        started = time.perf_counter()
        probabilities = torch.softmax(model(inputs), dim=1)
        latencies.append((time.perf_counter() - started) * 1000 / len(inputs))
        targets_all.append(targets.numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    targets = np.concatenate(targets_all)
    probabilities = np.concatenate(probabilities_all)
    predictions = probabilities.argmax(axis=1)
    chosen = probabilities[np.arange(len(targets)), targets]
    cross_entropy_bits = float(-np.log2(np.clip(chosen, 1e-12, 1)).mean())
    classes = probabilities.shape[1]
    result = {
        "accuracy": float((predictions == targets).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "cross_entropy_bits_per_target": cross_entropy_bits,
        "cross_entropy_bits_per_bit": cross_entropy_bits / model.target_bits,
        "perplexity": float(2 ** cross_entropy_bits),
        "brier_score": float(np.mean(np.sum((probabilities - np.eye(classes)[targets]) ** 2, axis=1))),
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "predictive_min_entropy_per_bit": float(-math.log2(max(probabilities.max(axis=1).mean(), 1e-12)) / model.target_bits),
        "mean_latency_ms_per_sample": float(np.mean(latencies)),
        "samples": int(len(targets)),
    }
    if classes == 2 and len(np.unique(targets)) == 2:
        result["roc_auc"] = float(roc_auc_score(targets, probabilities[:, 1]))
    return result
