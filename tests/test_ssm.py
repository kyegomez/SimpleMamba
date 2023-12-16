import torch
import pytest
from simple_mamba.ssm import SSM


@pytest.fixture
def ssm():
    return SSM(vocab_size=100, dim=50, state_dim=30, depth=2)


def test_ssm_init(ssm):
    assert isinstance(ssm, SSM)
    assert ssm.vocab_size == 100
    assert ssm.dim == 50
    assert ssm.state_dim == 30
    assert ssm.depth == 2


def test_ssm_embed(ssm):
    assert isinstance(ssm.embed, torch.nn.Embedding)
    assert ssm.embed.num_embeddings == 100
    assert ssm.embed.embedding_dim == 50


def test_ssm_ssm_layers(ssm):
    assert isinstance(ssm.ssm_layers, torch.nn.ModuleList)
    assert len(ssm.ssm_layers) == 2


def test_ssm_fc_out(ssm):
    assert isinstance(ssm.fc_out, torch.nn.Linear)
    assert ssm.fc_out.in_features == 30
    assert ssm.fc_out.out_features == 100


@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.randint(100, (1,)),
        torch.randint(100, (10,)),
        torch.randint(100, (100,)),
    ],
)
def test_ssm_forward(ssm, input_tensor):
    output = ssm(input_tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == input_tensor.shape[0]
    assert output.shape[1] == 100


# Add more tests as needed
