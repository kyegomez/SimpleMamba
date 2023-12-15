[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Simple Mamba

## Install
`pip install simple-mamba`


## Usage
```python
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