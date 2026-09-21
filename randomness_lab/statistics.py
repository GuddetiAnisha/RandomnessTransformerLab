from __future__ import annotations

import math
import zlib

import numpy as np
from scipy.special import erfc, gammaincc


def _validate(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(values) < 8 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected at least 8 binary values")
    return values


def monobit_test(bits: np.ndarray) -> dict:
    bits = _validate(bits)
    statistic = abs(np.sum(2 * bits.astype(np.int16) - 1)) / math.sqrt(len(bits))
    return {"statistic": float(statistic), "p_value": float(erfc(statistic / math.sqrt(2)))}


def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> dict:
    bits = _validate(bits)
    count = len(bits) // block_size
    if count < 2:
        raise ValueError("Sequence is too short for requested block size")
    blocks = bits[:count * block_size].reshape(count, block_size)
    chi_square = 4 * block_size * np.sum((blocks.mean(axis=1) - 0.5) ** 2)
    return {"statistic": float(chi_square), "p_value": float(gammaincc(count / 2, chi_square / 2)),
            "blocks": count, "block_size": block_size}


def runs_test(bits: np.ndarray) -> dict:
    bits = _validate(bits)
    n = len(bits)
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return {"statistic": float("inf"), "p_value": 0.0, "runs": 0, "eligible": False}
    runs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    statistic = abs(runs - 2 * n * pi * (1 - pi)) / denominator
    return {"statistic": float(statistic), "p_value": float(erfc(statistic)),
            "runs": runs, "eligible": True}


def serial_pair_test(bits: np.ndarray) -> dict:
    bits = _validate(bits)
    pairs = bits[:-1] * 2 + bits[1:]
    counts = np.bincount(pairs, minlength=4)
    expected = len(pairs) / 4
    chi_square = np.sum((counts - expected) ** 2 / expected)
    return {"statistic": float(chi_square), "p_value": float(gammaincc(1.5, chi_square / 2)),
            "counts": counts.tolist()}


def lag_autocorrelation(bits: np.ndarray, lag: int = 1) -> dict:
    bits = _validate(bits).astype(float)
    if lag < 1 or lag >= len(bits):
        raise ValueError("lag must be between 1 and sequence length - 1")
    x, y = bits[:-lag], bits[lag:]
    if x.std() == 0 or y.std() == 0:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    z_score = abs(correlation) * math.sqrt(max(len(x) - 3, 1))
    return {"correlation": correlation, "p_value": float(erfc(z_score / math.sqrt(2))), "lag": lag}


def longest_run(bits: np.ndarray) -> int:
    bits = _validate(bits)
    changes = np.flatnonzero(np.diff(bits) != 0) + 1
    lengths = np.diff(np.concatenate(([0], changes, [len(bits)])))
    return int(lengths.max())


def most_common_min_entropy(bits: np.ndarray) -> float:
    bits = _validate(bits)
    p_max = max(float(bits.mean()), float(1 - bits.mean()))
    return float(-math.log2(max(p_max, 1e-12)))


def markov_min_entropy(bits: np.ndarray) -> float:
    bits = _validate(bits)
    transitions = np.ones((2, 2), dtype=float)  # Laplace smoothing
    for left, right in zip(bits[:-1], bits[1:]):
        transitions[left, right] += 1
    probabilities = transitions / transitions.sum(axis=1, keepdims=True)
    return float(-math.log2(probabilities.max()))


def empirical_shannon_entropy(bits: np.ndarray) -> float:
    bits = _validate(bits)
    p = float(bits.mean())
    return float(sum(-q * math.log2(q) for q in (p, 1 - p) if q > 0))


def compression_ratio(bits: np.ndarray) -> float:
    bits = _validate(bits)
    packed = np.packbits(bits).tobytes()
    return float(len(zlib.compress(packed, level=9)) / max(len(packed), 1))


def analyze(bits: np.ndarray, alpha: float = 0.01) -> dict:
    bits = _validate(bits)
    tests = {
        "monobit": monobit_test(bits),
        "block_frequency": block_frequency_test(bits, min(128, max(8, len(bits) // 10))),
        "runs": runs_test(bits),
        "serial_pair": serial_pair_test(bits),
        "autocorrelation_lag_1": lag_autocorrelation(bits, 1),
        "autocorrelation_lag_8": lag_autocorrelation(bits, min(8, len(bits) - 1)),
    }
    return {
        "num_bits": len(bits),
        "proportion_ones": float(bits.mean()),
        "shannon_entropy_per_bit": empirical_shannon_entropy(bits),
        "mcv_min_entropy_per_bit": most_common_min_entropy(bits),
        "markov_min_entropy_per_bit": markov_min_entropy(bits),
        "compression_ratio": compression_ratio(bits),
        "longest_run": longest_run(bits),
        "tests": tests,
        "failed_tests": [name for name, result in tests.items() if result.get("p_value", 1.0) < alpha],
        "alpha": alpha,
    }
