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

if TYPE_CHECKING:
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

    def add_request(
        self,
        prompt: str | list[int],
        max_tokens: int = 128,
    ) -> Sequence:
        seq = Sequence(self.tokenizer.tokenizer(prompt), block_size = self.config.block_size)
        self.scheduler.add(seq)

    def step(self) -> list[tuple[int, list[int]]]:
        """执行一轮：

        seqs, is_prefill = scheduler.schedule()
        token_ids = model_runner.run(seqs, is_prefill)
        scheduler.postprocess(seqs, token_ids, is_prefill)

        只返回本轮新完成的 (seq_id, completion_token_ids)。
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Scheduler 中没有 waiting 和 running 的序列时返回 True。"""
        raise NotImplementedError

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        max_tokens: int = 128,
    ) -> list[str]:
        """同步接口：提交全部请求，循环 step，最后按输入顺序解码返回。"""
        raise NotImplementedError

    def exit(self) -> None:
        """释放 ModelRunner、进程等资源；应保证可重复调用。"""
        raise NotImplementedError
