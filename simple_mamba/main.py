import torch
from einops import rearrange
from torch import einsum, nn

# helpers
class MambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        heads,
        in_channels: int = None,
        out_channels: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Project to hidden dim
        self.proj = nn.Linear(dim, dim)

        # Silu activation
        self.silu = nn.SiLU()

        # Conv1d
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            *args,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        x, x_ = self.proj(x)

        # Apply silu to
        x_ = self.silu(x_)

        # Apply conv1d to x
        x = self.conv(x)

        x = self.silu(x)

        # Apply ssm to x
        x = ssm(x)

        # Mat mul with x_
        x = x @ x_.transpose(-1, -2)

        # Final Proj to dim
        x = self.proj(x)

        return x


# Transformer


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ParallelTransformerBlock(
                    dim, dim_head, heads, ff_mult
                ),
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


# classes


class BitNetTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, ff_mult
        )

        self.to_logits = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)
