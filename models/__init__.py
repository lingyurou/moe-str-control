import torch
import torch.nn as nn
from .backbone import VisualBackbone
from .moe_layer import TokenSequenceMoELayer
from .seq_head import CTCHead

class GMoESTRModel(nn.Module):
    """G-MoE-STR 全体整合端到端模型"""
    def __init__(self, num_classes=96, d_model=256, num_experts=4, top_k=1):
        super().__init__()
        self.backbone = VisualBackbone(d_model)
        self.moe_block = TokenSequenceMoELayer(d_model, num_experts, top_k)
        self.head = CTCHead(d_model, num_classes)

    def forward(self, images):
        # 1. 提取视觉序列
        features = self.backbone(images) # [B, T, d_model]
        # 2. 扔进门控专家混合网络层
        moe_features, aux_loss = self.moe_block(features)
        # 3. 产生字符预测概率
        logits = self.head(moe_features) # [T, B, num_classes]
        return logits, aux_loss