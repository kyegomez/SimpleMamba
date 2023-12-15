import pytest
import torch
from simple_mamba.main import MambaBlock


def test_mamba_block_output_shape():
    mamba_block = MambaBlock(
        dim=512,
        hidden_dim=128,
        heads=8,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
    )
    x = torch.randn(1, 3, 512)
    output = mamba_block(x)
    assert output.shape == x.shape


def test_mamba_block_output_type():
    mamba_block = MambaBlock(
        dim=512,
        hidden_dim=128,
        heads=8,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
    )
    x = torch.randn(1, 3, 512)
    output = mamba_block(x)
    assert isinstance(output, torch.Tensor)


# Repeat similar tests for different scenarios
# ...


# Test 20
def test_mamba_block_output_values():
    mamba_block = MambaBlock(
        dim=512,
        hidden_dim=128,
        heads=8,
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
    )
    x = torch.randn(1, 3, 512)
    output = mamba_block(x)
    assert torch.all(
        output >= 0
    )  # assuming output should be non-negative
