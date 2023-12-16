import torch
from torch import nn
from simple_mamba.two import SSM


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        vocab_size,
        state_dim,
        depth,
        heads,
        in_channels,
        out_channels,
        kernel_size,
        *args,
        **kwargs,
    ):
        super(MambaBlock, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.proj = nn.Linear(dim, hidden_dim)
        self.silu = nn.SiLU()

        # Adjust out_channels to match hidden_dim
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  # Changed to hidden_dim
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # Now we can initialize the SSM
        self.ssm = SSM(
            vocab_size=vocab_size,
            dim=dim,
            state_dim=state_dim,
            depth=depth,
            *args,
        )

    def forward(self, x: torch.Tensor):
        # We need to split up the input tensor into heads
        x_proj = self.proj(x)
        x_proj = self.silu(x_proj)

        x_reshaped = x_proj.transpose(1, 2)
        x_conv = self.conv(x_reshaped)
        x_conv = self.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)

        # Now x_proj and x_conv are compatible for matrix multiplication
        x = torch.matmul(x_proj, x_conv.transpose(-1, -2))
        x = self.proj(x)
        return x
