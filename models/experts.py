import torch
import torch.nn as nn

class FeedForwardExpert(nn.Module):
    """MoE 的独立专家网络 (标准两层 MLP FFN)"""
    def __init__(self, d_model, d_ffn=1024):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))