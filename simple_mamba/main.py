import torch
from einops import rearrange
from torch import einsum, nn


# helpers
class MambaBlock(nn.Module):
    """MambaBlock

    Args:
        nn (_type_): _description_


    Attributes:
        dim (int): _description_
        hidden_dim (int): _description_
        heads (int): _description_
        proj (nn.Linear): _description_
        silu (nn.SiLU): _description_
        conv (nn.Conv1d): _description_

    Example:
        >>> import torch
        >>> from simple_mamba.main import MambaBlock
        >>> mamba_block = MambaBlock(dim=512, hidden_dim=128, heads=8, in_channels=3, out_channels=3, kernel_size=3, stride=1)
        >>> x = torch.randn(1, 3, 512)
        >>> output = mamba_block(x)
        >>> assert output.shape == x.shape

    """

    def __init__(
        self,
        dim,
        hidden_dim,
        heads,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Project to hidden dim
        self.proj = nn.Linear(dim, dim)

        # Silu activation
        self.silu = nn.SiLU()

        # Conv1d
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        # ... [previous operations] ...

        x = self.proj(x)
        x_ = self.proj(x)

        # Apply silu
        x_ = self.silu(x_)

        # Apply conv1d and silu to x
        # x = self.conv(x)
        x = self.silu(x)

        # Reshape x_ to match x for matrix multiplication
        x_ = x_.view(
            x.shape[0], x.shape[1], -1
        )  # Reshape to [1, 3, 510]

        # Matrix multiplication
        x = torch.matmul(x, x_.transpose(-1, -2))

        # Final Proj to dim
        x = self.proj(x)

        return x


# Transformer


# class Transformer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         depth,
#         heads,
#         dim_head,
#         ff_mult=4,
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([])

#         for _ in range(depth):
#             self.layers.append(
#                 ParallelTransformerBlock(
#                     dim, dim_head, heads, ff_mult
#                 ),
#             )

#     def forward(self, x):
#         for block in self.layers:
#             x = block(x) + x
#         return x


# # classes


# class BitNetTransformer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         depth,
#         num_tokens,
#         dim_head=64,
#         heads=8,
#         ff_mult=4,
#     ):
#         super().__init__()
#         self.emb = nn.Embedding(num_tokens, dim)

#         self.transformer = Transformer(
#             dim, depth, heads, dim_head, ff_mult
#         )

#         self.to_logits = nn.Sequential(
#             RMSNorm(dim), nn.Linear(dim, num_tokens)
#         )

#     def forward(self, x):
#         x = self.emb(x)
#         x = self.transformer(x)
#         return self.to_logits(x)
