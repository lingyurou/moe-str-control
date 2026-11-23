import torch
import torch.nn as nn

class CTCHead(nn.Module):
    """文本序列分类解码头"""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 输入: [B, T, d_model]
        logits = self.linear(x) # [B, T, num_classes]
        # CTC 损失函数要求输入格式为 [T, B, num_classes]
        return logits.permute(1, 0, 2)