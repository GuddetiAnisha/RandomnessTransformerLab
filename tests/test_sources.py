import numpy as np
import pytest

from randomness_lab.sources import bytes_to_bits, generate, markov, periodic


def test_bytes_to_bits_known_value():
    assert bytes_to_bits(bytes([0b10100000]), 4).tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize("source", ["numpy", "mt", "lcg", "biased", "markov",
                                          "periodic", "hash_drbg_style", "hmac_drbg_style"])
def test_generators_are_binary_and_sized(source):
    bits = generate(source, 257, seed=7)
    assert bits.shape == (257,)
    assert set(np.unique(bits)) <= {0, 1}


def test_seeded_generator_reproducible():
    assert np.array_equal(markov(100, seed=5), markov(100, seed=5))


def test_periodic_pattern():
    assert periodic(8, "01").tolist() == [0, 1] * 4
