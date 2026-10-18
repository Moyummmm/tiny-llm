from dataclasses import dataclass


@dataclass
class Config:
    gpu_utilization: float = 0.7
    block_size: int = 4
