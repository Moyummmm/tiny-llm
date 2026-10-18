"""Tokenizer API used by LLMEngine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """LLMEngine 依赖的最小 tokenizer 接口。"""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """文本 -> token ids。"""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """token ids -> 文本。"""
        ...


class CharTokenizer:
    """字符级 tokenizer，仅用于本地开发和测试，不依赖 transformers。"""

    def __init__(self, vocab: list[str] | None = None):
        self.vocab = list(vocab) if vocab is not None else []
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.eos_token_id = self.token_to_id.get("<eos>")

    def _token_id(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.vocab)
            self.vocab.append(token)
        return self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        return [self._token_id(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.vocab[token_id]
            for token_id in token_ids
            if 0 <= token_id < len(self.vocab)
            and token_id != self.eos_token_id
        )


class HFTokenizer:
    """HuggingFace tokenizer 包装，后续接入真实模型时实现。"""

    def __init__(self, model_path: str, use_fast: bool = True):
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_fast: bool = True,
    ) -> "HFTokenizer":
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int | None:
        raise NotImplementedError
