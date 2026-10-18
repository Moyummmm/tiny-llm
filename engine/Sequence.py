from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    def __init__(
        self,
        seq_id: int,
        token_ids: list[int],
        block_size: int = 4,
        max_tokens: int = 128,
    ):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = self.num_tokens
        self.max_tokens = max_tokens

        # 已计算完 KV、且对应块已在 block_table 中的 token 数
        self.num_cached_tokens = 0
        # 本次调度要处理的 token 数
        self.num_scheduled_tokens = 0

        # 逻辑块下标 -> 物理 block_id
        self.block_table: list[int] = []
        self.block_size = block_size
        self.status = SequenceStatus.WAITING

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, index):
        return self.token_ids[index]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_blocks == 0:
            return 0
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def completion_token_ids(self) -> list[int]:
        """只返回生成出来的部分，不含 prompt。"""
        return self.token_ids[self.num_prompt_tokens:]

    def block(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.num_tokens += 1
