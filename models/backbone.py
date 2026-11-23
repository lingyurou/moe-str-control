import torch
import torch.nn as nn

class VisualBackbone(nn.Module):
    """视觉特征提取主干网络 (精简版 SVTR/CRNN 风格)"""
    def __init__(self, d_model=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # [64, 16, 64]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # [128, 8, 32]
            nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 25)) # 强行将宽压制成序列长度 T=25, 高 H=1
        )

    def forward(self, x):
        # 输入: [B, 3, 32, 128]
        x = self.features(x) # [B, d_model, 1, 25]
        x = x.squeeze(2)     # [B, d_model, 25]
        x = x.permute(0, 2, 1) # 转换成序列维度 [B, 25, d_model]
        return x