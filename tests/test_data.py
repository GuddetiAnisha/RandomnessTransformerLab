import numpy as np

from randomness_lab.data import BinarySequenceDataset, block_to_class, contiguous_split


def test_block_encoding():
    assert block_to_class(np.array([1, 0, 1])) == 5


def test_dataset_context_and_target():
    dataset = BinarySequenceDataset(np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8), 3, 2)
    context, target = dataset[0]
    assert context.tolist() == [0, 1, 1]
    assert target.item() == 1


def test_contiguous_split_has_no_overlap():
    bits = np.arange(100)
    train, validation, test = contiguous_split(bits)
    assert len(train) == 70 and len(validation) == 15 and len(test) == 15
    assert train[-1] < validation[0] < test[0]
