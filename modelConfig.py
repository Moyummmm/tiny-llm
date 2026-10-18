from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    dtype: torch.dtype = torch.float16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
