import torch
from simple_mamba.main import MambaBlock

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
