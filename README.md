[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Simple Mamba

## Install
`pip install simple-mamba`


## Usage
```python
import torch
from simple_mamba import MambaBlock


# Define block parameters
dim = 512
hidden_dim = 128
heads = 8
in_channels = 3
out_channels = 3
kernel_size = 3

# Create an instance of MambaBlock
mamba_block = MambaBlock(
    dim, hidden_dim, heads, in_channels, out_channels, kernel_size
)

# Create a sample input tensor
x = torch.randn(1, dim, dim)

# Pass the tensor through the MambaBlock
output = mamba_block(x)
print("Output shape:", output.shape)


```


# License
MIT


# Citation
```bibtex
@misc{gu2023mamba,
    title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces}, 
    author={Albert Gu and Tri Dao},
    year={2023},
    eprint={2312.00752},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

```