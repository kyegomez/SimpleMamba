import torch
from torch import nn
import torch.nn.functional as F


class SSM(nn.Module):
    """SSm is a state-space model for language modeling.

    Args:
        vocab_size (int): Size of the vocabulary.
        dim (int): Dimensionality of the embedding.
        state_dim (int): Dimensionality of the state.
        depth (int): Number of state-space layers.

    Examples:
        >>> ssm = SSM(vocab_size=100, dim=50, state_dim=30, depth=2)
        >>> input_tensor = torch.randint(100, (10,))
        >>> output = ssm(input_tensor)
        >>> output.shape
        torch.Size([10, 100])

    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        state_dim: int,
        depth: int,
        *args,
        **kwargs,
    ):
        super(SSM, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.state_dim = state_dim
        self.depth = depth

        self.embed = nn.Embedding(vocab_size, dim)
        self.ssm_layers = nn.ModuleList(
            [
                StateSpaceLayer(
                    dim if i == 0 else state_dim, state_dim
                )
                for i in range(depth)
            ]
        )

        self.fc_out = nn.Linear(state_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SSM.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        embeds = self.embed(x)
        state = embeds

        for layer in self.ssm_layers:
            state = layer(state)

        logits = self.fc_out(state)
        return logits


class StateSpaceLayer(nn.Module):
    def __init__(self, input_dim, state_dim):
        """
        A single layer of the state-space model for language modeling.

        Args:
        input_dim (int): Dimensionality of the input vector.
        state_dim (int): Dimensionality of the state vector.

        This layer uses linear transformations to model the state transitions.
        """
        super(StateSpaceLayer, self).__init__()
        self.state_transform = nn.Linear(input_dim, state_dim)
        self.output_transform = nn.Linear(state_dim, state_dim)

    def forward(self, x):
        """
        Forward pass through the State-Space Layer.

        Args:
        x (torch.Tensor): Input tensor for the layer.

        Returns:
        torch.Tensor: The transformed state.
        """
        state = F.relu(self.state_transform(x))
        output = self.output_transform(state)
        return output


# # Example usage
# vocab_size = 10000  # Example vocabulary size
# embed_dim = 256  # Example embedding dimension
# state_dim = 512  # State dimension
# num_layers = 2  # Number of state-space layers

# model = SSM(vocab_size, embed_dim, state_dim, num_layers)

# # Example input (sequence of word indices)
# input_seq = torch.randint(
#     0, vocab_size, (32, 10)
# )  # Batch size of 32, sequence length of 10

# # Forward pass
# logits = model(input_seq)
# print(logits.shape)  # Should be [32, 10, vocab_size]
