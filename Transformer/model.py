from Transformer.component import EncoderBlock, DecoderBlock,PositionalEncoding, Embedding

import torch.nn as nn


class Transformer(nn.Module):
    """论文《Attention Is All You Need》的 base 结构。

    参数
    ----
    N               : 编码器/解码器层数（base=6）
    heads           : 注意力头数（base=8）
    dims            : d_model（base=512）
    des_vocab_size  : 词表大小（本项目 37000，en/de 共享）
    tie_weights     : 是否做 weight tying（论文 §3.4 要求做）
    """

    def __init__(self, N, heads, dims, des_vocab_size, tie_weights=True):
        super().__init__()
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(heads, dims) for _ in range(N)])
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(heads, dims) for _ in range(N)])

        # 注意参数顺序：Embedding(vocab_size, dims)
        self.des_vocab_size = des_vocab_size
        self.Embedding = Embedding(des_vocab_size, dims)
        self.PE = PositionalEncoding(dims)
        self.Dropout = nn.Dropout(0.1)

        self.Linear = nn.Linear(dims, des_vocab_size, bias=False)

        # weight tying：输出投影不再单独拥有权重，直接指向 embedding 的权重矩阵。
        if tie_weights:
            self.Linear.weight = self.Embedding.Embedding.weight

    def forward(self, src, tgt_in, src_pad_mask=None, tgt_pad_mask=None):
        src_x = self.Dropout(self.PE(self.Embedding(src)))
        tgt_x = self.Dropout(self.PE(self.Embedding(tgt_in)))

        for blk in self.EncoderBlocks:
            src_x = blk(src_x, src_pad_mask)

        for blk in self.DecoderBlocks:
            tgt_x = blk(tgt_x, src_x, tgt_pad_mask, src_pad_mask)

        return self.Linear(tgt_x)  # (B, T, vocab_size) 的 logits



