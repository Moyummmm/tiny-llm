"""ModelRunner：把 Scheduler 选出的 batch 变成一次前向 + 一次采样。

职责：
1. 准备输入张量（input_ids / positions / slot_mapping / block_tables）
2. 分配 KV cache 并绑定到各层
3. 调用模型前向拿到 logits
4. greedy 采样出下一个 token

不负责：请求排队、块分配策略、tokenizer。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import Config
from ..modelConfig import ModelConfig
from .block_manager import BlockPool
from .Sequence import Sequence


@dataclass
class BatchInputs:
    """一次前向所需的全部输入和 attention 元数据。"""

    input_ids: torch.Tensor      # [num_tokens]
    positions: torch.Tensor      # [num_tokens]
    slot_mapping: torch.Tensor   # [num_tokens] 每个 token 写入 KV cache 的位置
    block_tables: torch.Tensor   # [num_seqs, max_blocks] 逻辑块 -> 物理块
    cu_seqlens_q: torch.Tensor   # [num_seqs + 1] 各 seq 本轮 query 的累计长度
    cu_seqlens_k: torch.Tensor   # [num_seqs + 1] 各 seq 本轮 key 的累计长度
    max_seqlen_q: int
    max_seqlen_k: int
    is_prefill: bool


class ModelRunner:
    """执行层：只关心“这批 seq 怎么算”，不决定“算谁”。"""

    def __init__(
        self,
        config: Config,
        model_config: ModelConfig,
        block_pool: BlockPool,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.block_pool = block_pool
        self.block_size = config.block_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.kv_cache: torch.Tensor | None = None
        self._closed = False

        if model is not None:
            self.kv_cache = self._allocate_kv_cache()
            self._bind_kv_cache()

    # ---------- 对外接口 ----------

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """跑一次前向，返回每条 seq 采样出的下一个 token。"""
        if not seqs:
            return []

        batch = self.prepare_inputs(seqs, is_prefill)
        logits = self._forward(batch)
        return self.sample(logits, seqs)

    def exit(self) -> None:
        """释放权重、KV cache 和显存；可重复调用。"""
        if self._closed:
            return
        self._closed = True

        self.model = None
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ---------- 输入准备 ----------

    def prepare_inputs(self, seqs: list[Sequence], is_prefill: bool) -> BatchInputs:
        """把 batch 摊平成一次 varlen 前向所需的张量。

        prefill 和 decode 走同一套逻辑：
        - prefill: 每个 seq 贡献 num_scheduled_tokens 个 token
        - decode:  每个 seq 贡献 1 个 token（就是还没算 KV 的那个）
        """
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(
                seq.block_table[t // self.block_size] * self.block_size
                + t % self.block_size
                for t in range(start, end)
            )

            cu_seqlens_q.append(cu_seqlens_q[-1] + seq.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, seq.num_scheduled_tokens)
            max_seqlen_k = max(max_seqlen_k, end)

        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in seqs
        ]

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=self.device)

        return BatchInputs(
            input_ids=tensor(input_ids, torch.long),
            positions=tensor(positions, torch.long),
            slot_mapping=tensor(slot_mapping, torch.long),
            block_tables=tensor(block_tables, torch.int32),
            cu_seqlens_q=tensor(cu_seqlens_q, torch.int32),
            cu_seqlens_k=tensor(cu_seqlens_k, torch.int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_prefill=is_prefill,
        )

    # ---------- 前向与采样 ----------

    def _forward(self, batch: BatchInputs) -> torch.Tensor:
        """调用模型，返回 [num_tokens, vocab_size] 的 logits。

        模型需要约定：
        - 从 input_ids/positions 算 hidden states
        - attention 层按 slot_mapping 写入 KV cache
        - 用 block_tables + cu_seqlens_* 找到每条 seq 的历史 KV
        """
        if self.model is None:
            raise RuntimeError("ModelRunner.model is None：请把模型实例传进来")

        return self.model(
            input_ids=batch.input_ids,
            positions=batch.positions,
            slot_mapping=batch.slot_mapping,
            block_tables=batch.block_tables,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=batch.max_seqlen_k,
            is_prefill=batch.is_prefill,
        )

    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """greedy：取每条 seq 最后一个 scheduled token 的 argmax。"""
        offsets: list[int] = []
        total = 0
        for seq in seqs:
            total += seq.num_scheduled_tokens
            offsets.append(total - 1)

        return logits[offsets].argmax(dim=-1).tolist()

    # ---------- KV cache ----------

    def _allocate_kv_cache(self) -> torch.Tensor:
        """形状 [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]。"""
        mc = self.model_config
        return torch.empty(
            2,
            mc.num_hidden_layers,
            self.block_pool.num_blocks,
            self.block_size,
            mc.num_key_value_heads,
            mc.head_dim,
            dtype=mc.dtype,
            device=self.device,
        )

    def _bind_kv_cache(self) -> None:
        """把每层的 k_cache / v_cache 指向 kv_cache 的对应切片。"""
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        self.num_bound_layers = layer_id
