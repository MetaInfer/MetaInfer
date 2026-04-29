"""前缀块共享：相同整块 token 序列复用物理块 id（引用计数）。"""

from engine.block_manager import BlockManager
from engine.structs import Sequence


def test_two_sequences_share_full_block_hash():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = Sequence(request_id="a", input_ids=[1, 2, 3, 4])
    a.block_size = 4
    b = Sequence(request_id="b", input_ids=[1, 2, 3, 4])
    b.block_size = 4

    bm.allocate(a)
    bm.allocate(b)
    assert a.block_table == b.block_table
    assert bm.blocks[a.block_table[0]].ref_count == 2

    bm.deallocate(a)
    assert bm.blocks[b.block_table[0]].ref_count == 1
    bm.deallocate(b)
