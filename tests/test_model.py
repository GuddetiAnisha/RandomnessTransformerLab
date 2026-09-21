import torch

from randomness_lab.model import CompactBitTransformer


def test_model_output_shape_and_parameters():
    model = CompactBitTransformer(context_length=16, target_bits=3, d_model=16,
                                  nhead=4, num_layers=1, dim_feedforward=32)
    logits = model(torch.randint(0, 2, (4, 16)))
    assert logits.shape == (4, 8)
    assert model.parameter_count > 0
