import torch
import torch.nn as nn

class CombinedMoELoss(nn.Module):
    """联合损失函数: STR 识别损失 (CTC) + MoE 路由均衡损失"""
    def __init__(self, alpha_balance=0.01):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.alpha_balance = alpha_balance

    def forward(self, log_probs, targets, input_lengths, target_lengths, aux_loss):
        # log_probs: [T, B, num_classes]
        # 计算传统文本识别损失
        loss_str = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        # 结合 MoE 的负载均衡损失
        total_loss = loss_str + self.alpha_balance * aux_loss
        return total_loss, loss_str