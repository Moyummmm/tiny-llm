"""LLMEngine：编排层。

职责：
1. 持有 tokenizer、Scheduler、ModelRunner
2. 接收请求，构造 Sequence 并交给 Scheduler
3. 驱动 schedule -> forward -> postprocess 循环
4. 收集已完成请求并解码
5. 释放模型和进程资源

不负责：块分配策略、batching 策略、模型前向细节。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .Sequence import Sequence
from .scheduler import Scheduler
from .model_runner import ModelRunner
from ..config import Config

class LLMEngine:
    def __init__(
        self,
        tokenizer,
        scheduler: Scheduler,
        model_runner: ModelRunner,
        config: Config
    ) -> None:
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.config = config
        self._next_seq_id = 0
        self._closed = False

    def add_request(
        self,
        prompt: str | list[int],
        max_tokens: int = 128,
    ) -> Sequence:
        token_ids = (
            self.tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        seq = Sequence(
            seq_id=self._next_seq_id,
            token_ids=token_ids,
            block_size=self.config.block_size,
            max_tokens=max_tokens,
        )
        self._next_seq_id += 1
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[tuple[int, list[int]]]:
        """跑一轮：调度 -> 前向 -> 收尾，返回本轮新完成的请求。"""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []

        token_ids = self.model_runner.run(seqs, is_prefill)
        finished = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return [(seq.seq_id, seq.completion_token_ids) for seq in finished]

    def is_finished(self) -> bool:
        """Scheduler 中没有 waiting 和 running 的序列时返回 True。"""
        return not self.scheduler.has_unfinished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int = 128,
    ) -> list[str]:
        """同步接口：提交全部请求，循环 step，最后按输入顺序解码返回。"""
        seq_ids = [self.add_request(p, max_tokens).seq_id for p in prompts]

        outputs: dict[int, list[int]] = {}
        while not self.is_finished():
            for seq_id, token_ids in self.step():
                outputs[seq_id] = token_ids

        # 完成顺序和提交顺序不同，必须按 seq_id 还原成输入顺序
        return [self.tokenizer.decode(outputs[seq_id]) for seq_id in seq_ids]

    def exit(self) -> None:
        """释放 ModelRunner、进程等资源；应保证可重复调用。"""
        if self._closed:
            return
        self._closed = True

        # 先归还 KV cache 块，再让 runner 释放权重/进程
        self.scheduler.clear()
        self.model_runner.exit()

        # 断开引用，让 tokenizer、模型和调度器可以被回收
        self.tokenizer = None
        self.scheduler = None
        self.model_runner = None
