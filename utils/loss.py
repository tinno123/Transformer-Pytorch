import torch
import torch.nn.functional as F
from torch import nn



class Loss(nn.Module):
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        """
        vocab_size  : 词表大小（本项目 37000）
        padding_idx : <pad> 的 id，这些位置不计入 loss
        smoothing   : 论文里的 ε_ls，base 模型取 0.1；填 0 就退回普通交叉熵
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, logits, target):
        B = logits.size(0)
        T = logits.size(1)
        if logits.dim() != 3:
            raise ValueError(f"logits 期望 (B,T,V) 收到 {tuple(logits.shape)}")

        # ------------------------------------------
        # 1. Reshape成 (batch × 序列长度, vocab_size) = (B*T,vocab_size)
        # ------------------------------------------
        logits = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        # ------------------------------------------
        # 2.softmax
        # ------------------------------------------
        log_p = F.log_softmax(logits, dim=-1)

        # ------------------------------------------
        # 3.生成One-Hot并进行标签平滑
        # ------------------------------------------
        #3.1 One-Hot
        label = torch.zeros_like(logits)
        label[torch.arange(B*T,device=logits.device), target] = 1
        # 3.2 平滑 1处为 1-ε
        label = (1.0 - self.smoothing) * label + self.smoothing / self.vocab_size

        # ------------------------------------------
        # 4. 去除 <pad> 位置的 loss
        # ------------------------------------------
        positive_mask = torch.where(target != self.padding_idx)[0].to(logits.device)
        log_p = log_p[positive_mask,:]
        label = label[positive_mask,:]

        # ------------------------------------------
        # 5. 计算损失 按Token进行平均
        # ------------------------------------------
        loss = -(label * log_p).sum(dim=-1)
        if loss.numel() == 0:
            return logits.sum() * 0.0

        return loss.mean()



