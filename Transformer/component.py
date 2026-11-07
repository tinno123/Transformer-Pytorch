import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):


    def __init__(self, heads=8, dims=512, Mask=False):
        super().__init__()
        self.heads = heads
        self.dims = dims
        self.d_k = dims // heads
        self.scale = math.sqrt(self.d_k)
        self.Mask = Mask

        self.q_proj = nn.Linear(self.dims, self.dims, bias=False)
        self.k_proj = nn.Linear(self.dims, self.dims, bias=False)
        self.v_proj = nn.Linear(self.dims, self.dims, bias=False)
        self.O_w = nn.Linear(self.dims, self.dims)

        self._causal_cache = {}

    def _causal_mask(self, S_q, S_k, device):

        need = max(S_q, S_k)
        cached = self._causal_cache.get(str(device))
        if cached is None or cached.size(0) < need:
            size = max(need, 128)     # 起步 128，避免频繁重建
            cached = torch.triu(torch.full((size, size), float("-inf"), device=device),
                                diagonal=1)
            self._causal_cache[str(device)] = cached
        return cached[:S_q, :S_k]

    def forward(self, Q_Input, K_Input, V_Input, key_padding_mask=None):
        B, S_q, _ = Q_Input.shape
        S_k = K_Input.size(1)
        H, d_k = self.heads, self.d_k

        # 1. 一次性投影，再 view 成多头： (B, S, dims) -> (B, heads, S, d_k)
        q = self.q_proj(Q_Input).view(B, S_q, H, d_k).transpose(1, 2)
        k = self.k_proj(K_Input).view(B, S_k, H, d_k).transpose(1, 2)
        v = self.v_proj(V_Input).view(B, S_k, H, d_k).transpose(1, 2)

        # 2. 缩放点积注意力，多头一次算完
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # (B, H, S_q, S_k)
        if self.Mask:
            scores = scores + self._causal_mask(S_q, S_k, scores.device)
        if key_padding_mask is not None:
            # (B, S_k) -> (B, 1, 1, S_k)，屏蔽的是 key 维，对所有 query 行和所有头共用
            scores = scores + key_padding_mask[:, None, None, :]
        attn = torch.matmul(scores.softmax(dim=-1), v)               # (B, H, S_q, d_k)

        # 3. 多头拼回 dims(本质上是reshape操作)
        attn = attn.transpose(1, 2).reshape(B, S_q, self.dims)
        return self.O_w(attn)



class FFN(nn.Module):
    def __init__(self, dims=512):
        super().__init__()
        self.dims = dims
        self.Linear1 = nn.Linear(self.dims, self.dims * 4)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(self.dims * 4, self.dims)

    def forward(self, x):
        return self.Linear2(self.ReLU(self.Linear1(x)))

class ResidualLNBlock(nn.Module):
    def __init__(self, dims=512):
        super().__init__()
        self.dims = dims
        self.LayerNorm = nn.LayerNorm(self.dims)
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x, res):
        return self.LayerNorm(self.Dropout(x)+res)



class PositionalEncoding(nn.Module):


    def __init__(self, dims=512, max_len=5000, base: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(base) / dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, : x.size(1)]

class Embedding(nn.Module):

    def __init__(self, vocab_size, dims):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size, dims)
        self.dims = dims
        nn.init.normal_(self.Embedding.weight, mean=0.0, std=dims ** -0.5)

    def forward(self, x):
        return self.Embedding(x) * (self.dims ** 0.5)






class EncoderBlock(nn.Module):
    def __init__(self, heads=8, dims=512):
        super().__init__()
        self.Attention = MultiHeadAttention(heads, dims, Mask=False)
        self.ResidualBlock1 = ResidualLNBlock(dims)
        self.FFN = FFN(dims)
        self.ResidualBlock2 = ResidualLNBlock(dims)

    def forward(self, x, key_padding_mask=None):
        h = self.ResidualBlock1(self.Attention(x, x, x, key_padding_mask), x)
        return self.ResidualBlock2(self.FFN(h), h)
class DecoderBlock(nn.Module):

    def __init__(self, heads=8, dims=512):
        super().__init__()
        self.MaskAttention = MultiHeadAttention(heads, dims, Mask=True)
        self.ResidualBlock1 = ResidualLNBlock(dims)
        self.Attention = MultiHeadAttention(heads, dims, Mask=False)
        self.ResidualBlock2 = ResidualLNBlock(dims)
        self.FFN = FFN(dims)
        self.ResidualBlock3 = ResidualLNBlock(dims)

    def forward(self, x, z, tgt_padding_mask=None, src_padding_mask=None):
        # 子层 1：带因果掩码的自注意力（Q=K=V=目标句）
        h = self.ResidualBlock1(self.MaskAttention(x, x, x, tgt_padding_mask), x)
        # 子层 2：交叉注意力，Q 来自 decoder，K/V 来自 encoder 输出 z
        h = self.ResidualBlock2(self.Attention(h, z, z, src_padding_mask), h)
        # 子层 3：FFN
        return self.ResidualBlock3(self.FFN(h), h)
