import torch
import hashlib
import numpy as np

from ..config import Config
from ..modelConfig import ModelConfig
from .Sequence import Sequence
from collections import deque

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash_value:int, token_ids:list[int]):
        self.hash = hash_value
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockPool():
    def __init__(self, conf: Config, model_config: ModelConfig):
        self.gpu_utilization = conf.gpu_utilization
        self.block_size = conf.block_size

        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        usable = total * self.gpu_utilization - used

        num_layers = model_config.num_hidden_layers
        num_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.head_dim

        kv_cache_byte = (
            2 * num_layers * num_kv_heads * head_dim
            * model_config.dtype.itemsize
        )
        self.num_blocks = int(usable // (kv_cache_byte * self.block_size))
        if self.num_blocks <= 0:
            raise RuntimeError("no GPU memory available for KV cache blocks")
        
        self.blocks = [Block(i) for i in range(self.num_blocks)]

        self.num_total: int = len(self.blocks)
        self.hashvalue_blockId_table: dict[int, int] = {}
        self.free_block = deque(range(self.num_blocks))  # 存 block_id
        self.used_block: set[int] = set()                 # 存 block_id
        self.num_heads = num_kv_heads
        self.head_dim = head_dim

    @property
    def num_free(self) -> int:
        return len(self.free_block)

    def can_allocate(self, num_new_blocks: int) -> bool:
        return self.num_free >= num_new_blocks

    def lookup(self, hash_value: int, token_ids: list[int]) -> int | None:
        """只查询前缀缓存，不修改任何状态。"""
        block_id = self.hashvalue_blockId_table.get(hash_value)
        if block_id is None:
            return None
        block = self.blocks[block_id]
        return block_id if block.token_ids == token_ids else None

    def allocate_cache(self, hash_value: int, token_ids: list[int]) -> int | None:
        """命中前缀缓存时认领该块，并增加引用计数。"""
        block_id = self.hashvalue_blockId_table.get(hash_value)
        if block_id is None:
            return None

        block = self.blocks[block_id]
        if block.token_ids != token_ids:  # 防 hash 碰撞
            return None

        if block_id in self.used_block:
            block.ref_count += 1
        else:
            self.free_block.remove(block_id)
            self.used_block.add(block_id)
            block.ref_count = 1

        return block_id

    def allocate_new(self, num: int = 1) -> list[int]:
        """分配 num 个全新的块，返回 block_id 列表。"""
        if num < 0:
            raise ValueError("num must be non-negative")
        if num > self.num_free:
            raise RuntimeError(f"need {num} blocks, only {self.num_free} free")

        block_ids = []
        for _ in range(num):
            block_id = self.free_block.popleft()
            block = self.blocks[block_id]
            assert block.ref_count == 0

            # 块将被新内容覆盖，删除旧 hash 映射
            if (
                block.hash != -1
                and self.hashvalue_blockId_table.get(block.hash) == block_id
            ):
                del self.hashvalue_blockId_table[block.hash]

            block.reset()  # ref_count=1，清空 hash/token_ids
            self.used_block.add(block_id)
            block_ids.append(block_id)

        return block_ids

    # ref_count减到零加到空闲队列
    def free(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"block {block_id} is already free"

        block.ref_count -= 1
        if block.ref_count == 0:
            self.used_block.remove(block_id)
            self.free_block.append(block_id)

    # 计算hash并设置
    def set_hash(self, block_id: int, hash_value: int, token_ids: list[int]) -> None:
        block = self.blocks[block_id]

        # 块被复用于新内容时，清掉旧的 hash 映射
        old_hash = block.hash
        if (
            old_hash != -1
            and old_hash != hash_value
            and self.hashvalue_blockId_table.get(old_hash) == block_id
        ):
            del self.hashvalue_blockId_table[old_hash]

        block.update(hash_value, token_ids)
        self.hashvalue_blockId_table[hash_value] = block_id

    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """计算带前缀依赖的块 hash：h_i = H(h_{i-1} || token_ids_i)。"""
        h = hashlib.blake2b(digest_size=8)
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little", signed=False))
        h.update(np.asarray(token_ids, dtype="<i8").tobytes())
        return int.from_bytes(h.digest(), "little")

    def get(self,block_id) -> Block:
        return self.blocks[block_id]



class BlockManager:

    def __init__(self, block_pool: BlockPool, config: Config):
        self.block_pool = block_pool
        self.config = config
        self.block_size = config.block_size

    def can_allocate(self, seq) -> int:
        """返回可复用的前缀块数；空闲块不足时返回 -1。"""
        if seq.block_table:
            raise RuntimeError(
                "can_allocate() is only for sequences without allocated blocks"
            )
        if seq.block_size != self.block_size:
            raise ValueError(
                f"block_size mismatch: seq={seq.block_size}, "
                f"manager={self.block_size}"
            )

        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        # 最后一个块可能是半块，不能作为可复用前缀
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.block_pool.compute_hash(token_ids, h)
            block_id = self.block_pool.lookup(h, token_ids)
            if block_id is None:
                break

            num_cached_blocks += 1
            # 已被其他 sequence 使用的块，不额外占用空闲块
            if block_id in self.block_pool.used_block:
                num_new_blocks -= 1

        if num_new_blocks > self.block_pool.num_free:
            return -1
        return num_cached_blocks

    def allocate(self, seq, num_cached_blocks: int = 0) -> None:
        """提交分配；调用前必须先用 can_allocate() 检查过容量。"""
        assert not seq.block_table, "sequence already allocated"
        assert seq.block_size == self.block_size
        if num_cached_blocks < 0 or num_cached_blocks > seq.num_blocks:
            raise ValueError("invalid num_cached_blocks")

        # 必须与 can_allocate() 使用相同的初始值和 hash 链
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.block_pool.compute_hash(token_ids, h)
            block_id = self.block_pool.allocate_cache(h, token_ids)
            if block_id is None:
                raise RuntimeError(
                    "prefix cache changed between can_allocate and allocate"
                )
            seq.block_table.append(block_id)

        num_new_blocks = seq.num_blocks - num_cached_blocks
        seq.block_table.extend(self.block_pool.allocate_new(num_new_blocks))
        seq.num_cached_tokens = num_cached_blocks * self.block_size
            

    def deallocate(self, seq) -> None:
        assert seq.block_table
        assert seq.block_size == self.block_size
        # 可能多出一块：prefill 后为采样出的 token 预留的块
        assert len(seq.block_table) >= seq.num_blocks

        for v in seq.block_table:
            self.block_pool.free(v)

        seq.block_table = []
        seq.num_cached_tokens = 0
        


    def can_append(self, seq) -> bool:
        assert seq.block_size == self.block_size
        if seq.num_blocks == 0:
            return self.block_pool.can_allocate(1)
        if seq.last_block_num_tokens < seq.block_size:
            return True
        else:
            return self.block_pool.can_allocate(1)

    def may_append(self, seq) -> None:
        if seq.num_blocks > 0 and seq.last_block_num_tokens < self.block_size:
            return
        
        seq.block_table.append(self.block_pool.allocate_new(1)[0])
        

    def hash_blocks(self, seq) -> None:
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) //self.block_size
        if start == end:    return

        h = -1 if start == 0 else self.block_pool.get(seq.block_table[start-1]).hash
        for i in range(start, end):
            block_id = seq.block_table[i]
            token_ids = seq.block(i)
            h = self.block_pool.compute_hash(token_ids, h)
            self.block_pool.set_hash(block_id, h, token_ids)



