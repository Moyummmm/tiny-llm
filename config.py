from dataclasses import dataclass


@dataclass
class Config:
    gpu_utilization: float = 0.7
    block_size: int = 4
    max_sequences: int = 8
    max_model_length = 1024
    max_num_batched_tokens = 8
