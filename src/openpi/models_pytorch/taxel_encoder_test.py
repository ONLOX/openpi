import pytest
import torch

from openpi.models_pytorch.taxel_encoder import TaxelEncoder
from openpi.models_pytorch.taxel_encoder import TaxelEncoderConfig
from openpi.models_pytorch.taxel_encoder import signed_log1p


@pytest.fixture
def small_encoder() -> TaxelEncoder:
    return TaxelEncoder(
        TaxelEncoderConfig(
            cnn_width=32,
            token_dim=64,
            transformer_layers=1,
            transformer_heads=4,
            transformer_mlp_dim=128,
        )
    )


def test_taxel_encoder_shape_and_gradient(small_encoder):
    force = torch.randn(2, 5, 7, 5, 3, requires_grad=True)
    tokens = small_encoder(force)

    assert tokens.shape == (2, 5, 64)
    tokens.square().mean().backward()
    assert force.grad is not None
    assert torch.isfinite(force.grad).all()


def test_taxel_encoder_rejects_wrong_shape(small_encoder):
    with pytest.raises(ValueError, match="expected taxel force shape"):
        small_encoder(torch.randn(2, 5, 35, 3))


def test_signed_log1p_preserves_sign_and_zero():
    value = torch.tensor([-3.0, 0.0, 3.0])
    transformed = signed_log1p(value, 1.0)

    torch.testing.assert_close(transformed[0], -transformed[2])
    assert transformed[1] == 0


def test_signed_log1p_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="force scale must be positive"):
        signed_log1p(torch.ones(1), 0)
