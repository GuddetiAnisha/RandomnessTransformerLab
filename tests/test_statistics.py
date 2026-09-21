import numpy as np

from randomness_lab.statistics import analyze, markov_min_entropy, monobit_test


def test_balanced_sequence_passes_monobit():
    bits = np.tile([0, 1], 1000).astype(np.uint8)
    assert monobit_test(bits)["p_value"] == 1.0


def test_biased_sequence_detected():
    bits = np.concatenate([np.ones(900), np.zeros(100)]).astype(np.uint8)
    result = analyze(bits)
    assert "monobit" in result["failed_tests"]
    assert result["mcv_min_entropy_per_bit"] < 0.2


def test_markov_entropy_detects_dependency():
    bits = np.repeat([0, 1], 500).astype(np.uint8)
    assert markov_min_entropy(bits) < 0.02
