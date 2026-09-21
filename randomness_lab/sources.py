from __future__ import annotations

import hashlib
import hmac
import os
import random
from pathlib import Path

import numpy as np


def bytes_to_bits(data: bytes, num_bits: int | None = None) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits[:num_bits].astype(np.uint8) if num_bits else bits.astype(np.uint8)


def _enough_bytes(num_bits: int) -> int:
    return (num_bits + 7) // 8


def os_random(num_bits: int) -> np.ndarray:
    return bytes_to_bits(os.urandom(_enough_bytes(num_bits)), num_bits)


def numpy_prng(num_bits: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, num_bits, dtype=np.uint8)


def mersenne_twister(num_bits: int, seed: int = 42) -> np.ndarray:
    rng = random.Random(seed)
    return np.fromiter((rng.getrandbits(1) for _ in range(num_bits)), dtype=np.uint8)


def lcg(num_bits: int, seed: int = 42, a: int = 1103515245, c: int = 12345,
        modulus: int = 2**31, output_bit: int = 4) -> np.ndarray:
    """Intentionally weak LCG source; output_bit controls how obvious the defect is."""
    state = seed % modulus
    bits = np.empty(num_bits, dtype=np.uint8)
    for idx in range(num_bits):
        state = (a * state + c) % modulus
        bits[idx] = (state >> output_bit) & 1
    return bits


def biased(num_bits: int, seed: int = 42, p_one: float = 0.60) -> np.ndarray:
    if not 0 <= p_one <= 1:
        raise ValueError("p_one must be between 0 and 1")
    return (np.random.default_rng(seed).random(num_bits) < p_one).astype(np.uint8)


def markov(num_bits: int, seed: int = 42, p_stay: float = 0.72) -> np.ndarray:
    if not 0 <= p_stay <= 1:
        raise ValueError("p_stay must be between 0 and 1")
    rng = np.random.default_rng(seed)
    bits = np.empty(num_bits, dtype=np.uint8)
    bits[0] = rng.integers(0, 2)
    for idx in range(1, num_bits):
        bits[idx] = bits[idx - 1] if rng.random() < p_stay else 1 - bits[idx - 1]
    return bits


def periodic(num_bits: int, pattern: str = "001011") -> np.ndarray:
    if not pattern or set(pattern) - {"0", "1"}:
        raise ValueError("pattern must be a non-empty binary string")
    raw = np.fromiter((int(ch) for ch in pattern), dtype=np.uint8)
    return np.resize(raw, num_bits)


def hash_drbg_style(num_bits: int, seed: int = 42) -> np.ndarray:
    """Counter-mode SHA-256 research source; not a validated SP 800-90A DRBG."""
    key = seed.to_bytes(32, "big", signed=False)
    output = bytearray()
    counter = 0
    while len(output) < _enough_bytes(num_bits):
        output.extend(hashlib.sha256(key + counter.to_bytes(16, "big")).digest())
        counter += 1
    return bytes_to_bits(bytes(output), num_bits)


def hmac_drbg_style(num_bits: int, seed: int = 42) -> np.ndarray:
    """HMAC-SHA256 research source; not a validated SP 800-90A DRBG."""
    key = hashlib.sha256(seed.to_bytes(32, "big", signed=False)).digest()
    output = bytearray()
    counter = 0
    while len(output) < _enough_bytes(num_bits):
        output.extend(hmac.new(key, counter.to_bytes(16, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes_to_bits(bytes(output), num_bits)


GENERATORS = {
    "os": os_random,
    "numpy": numpy_prng,
    "mt": mersenne_twister,
    "lcg": lcg,
    "biased": biased,
    "markov": markov,
    "periodic": periodic,
    "hash_drbg_style": hash_drbg_style,
    "hmac_drbg_style": hmac_drbg_style,
}


def generate(source: str, num_bits: int, seed: int = 42, **kwargs) -> np.ndarray:
    if num_bits < 8:
        raise ValueError("num_bits must be at least 8")
    if source not in GENERATORS:
        raise ValueError(f"Unknown source {source!r}; choose from {sorted(GENERATORS)}")
    function = GENERATORS[source]
    if source in {"os", "periodic"}:
        return function(num_bits, **kwargs)
    return function(num_bits, seed=seed, **kwargs)


def load_bits(path: str | Path, fmt: str = "raw", bit_column: str = "bit") -> np.ndarray:
    path = Path(path)
    if fmt == "raw":
        return bytes_to_bits(path.read_bytes())
    if fmt == "ascii":
        text = "".join(path.read_text(encoding="utf-8").split())
        if set(text) - {"0", "1"}:
            raise ValueError("ASCII input must contain only 0 and 1 plus whitespace")
        return np.fromiter((int(ch) for ch in text), dtype=np.uint8)
    if fmt == "hex":
        text = "".join(path.read_text(encoding="utf-8").split())
        return bytes_to_bits(bytes.fromhex(text))
    if fmt == "npy":
        values = np.load(path).reshape(-1)
    elif fmt == "csv":
        import pandas as pd
        values = pd.read_csv(path)[bit_column].to_numpy()
    else:
        raise ValueError("format must be raw, ascii, hex, npy, or csv")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Input values must be binary")
    return values.astype(np.uint8)
