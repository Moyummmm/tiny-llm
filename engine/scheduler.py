"""Scheduler：决定每一轮跑哪些 Sequence、每个跑多少 token。"""

from collections import deque

from ..config import Config
from .Sequence import Sequence, SequenceStatus
from .block_manager import BlockManager


class Scheduler:
    def __init__(
        self,
        config: Config,
        block_manager: BlockManager,
        eos_token_id: int | None = None,
    ):
        """初始化配置、块管理器和等待/运行队列。"""
        self.config = config
        self.block_manager = block_manager
        self.eos_token_id = eos_token_id
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """把新请求放入等待队列。"""
        assert not seq.block_table
        assert seq.num_cached_tokens == 0
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """挑出本轮要跑的 batch，并返回是否为 prefill 阶段。"""
        if self.running:
            return self._schedule_decode(), False
        if self.waiting:
            return self._schedule_prefill(), True
        return [], False

    def _schedule_prefill(self) -> list[Sequence]:
        """按 FIFO 并结合预算与块容量，从 waiting 中组成 prefill batch。"""
        batch: list[Sequence] = []
        budget = self.config.max_num_batched_tokens

        while self.waiting and len(batch) < self.config.max_sequences:
            seq = self.waiting[0]
            num_cached = self.block_manager.can_allocate(seq)
            if num_cached < 0:
                break

            num_scheduled = seq.num_tokens - num_cached
            # 单条超过预算时，只要是本轮第一条就放行，否则永远调度不到
            if batch and num_scheduled > budget:
                break

            self.waiting.popleft()
            self.block_manager.allocate(seq, num_cached)
            # prefill 后采样出的第一个 token 也要占位；prompt 正好填满整块时
            # 必须提前再要一块，否则 append 后块表会比 num_blocks 少一块
            if seq.num_tokens % self.config.block_size == 0:
                if not self.block_manager.can_append(seq):
                    self.block_manager.deallocate(seq)
                    self.waiting.appendleft(seq)
                    break
                self.block_manager.may_append(seq)
            seq.num_scheduled_tokens = num_scheduled
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            batch.append(seq)
            budget -= num_scheduled

        return batch

    def _schedule_decode(self) -> list[Sequence]:
        """为 running 中的每条序列各调度 1 个 token 组成 decode batch。"""
        batch: list[Sequence] = []
        for seq in list(self.running):
            if seq.is_finished:
                continue
            if self.block_manager.can_append(seq):
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                batch.append(seq)
                continue

            # 块不够：抢占，释放后放回队首，下轮重新 prefill
            self.block_manager.deallocate(seq)
            seq.status = SequenceStatus.WAITING
            seq.num_scheduled_tokens = 0
            self.running.remove(seq)
            self.waiting.appendleft(seq)
            break

        return batch

    def postprocess(self, seqs, token_ids, is_prefill) -> list[Sequence]:
        """回写本轮生成的 token，并返回本轮新完成的序列。"""
        if len(seqs) != len(token_ids):
            raise ValueError(
                f"batch size mismatch: {len(seqs)} seqs vs {len(token_ids)} tokens"
            )

        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_finished:
                continue

            # hash_blocks 依赖“旧 cached + 本轮 scheduled”，必须放在更新 cached 之前
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            seq.append_token(token_id)
            num_generated = seq.num_tokens - seq.num_prompt_tokens

            hit_eos = self.eos_token_id is not None and token_id == self.eos_token_id
            hit_limit = num_generated >= seq.max_tokens
            if not (hit_eos or hit_limit):
                continue

            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)
            finished.append(seq)

        return finished

    def has_unfinished(self) -> bool:
        """仍有等待或运行中的序列时返回 True。"""
        return bool(self.waiting or self.running)

    def clear(self) -> None:
        """丢弃所有请求并归还已占用的块。"""
        for seq in list(self.running):
            if seq.block_table:
                self.block_manager.deallocate(seq)
            seq.num_scheduled_tokens = 0
        self.waiting.clear()
        self.running.clear()
