import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSequenceMoELayer(nn.Module):
    """针对 STR 序列文本设计的 Token-level Top-k 门控专家混合层"""

    def __init__(self, d_model, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控路由网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # 实例化所有专家
        from .experts import FeedForwardExpert
        self.experts = nn.ModuleList([FeedForwardExpert(d_model) for _ in range(num_experts)])

    def forward(self, x):
        # 输入维度: [B, T, C] -> 代表 (Batch, Sequence_length, Channels)
        B, T, C = x.shape
        x_flat = x.contiguous().view(-1, C)  # 先深拷贝一份连续内存，再进行 view

        # 1. 计算门控路由 logits
        gate_logits = self.gate(x_flat)  # [B*T, num_experts]

        # 2. 计算路由权重与 Top-K 选择
        gate_softmax = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_softmax, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 重新归一化

        # 3. 计算用于辅助损失的负载均衡变量 (Load Balancing Loss Components)
        # 重要性 P: 路由概率均值
        importance = gate_softmax.mean(dim=0)
        # 负载 F: 各专家被选为 Top-1 的真实 Token 比例
        top1_indices = topk_indices[:, 0]
        load = torch.bincount(top1_indices, minlength=self.num_experts).float() / x_flat.size(0)
        # 负载均衡辅助损失 (完美的均衡状态下，由于 importance 和 load 一致，该损失最小)
        aux_loss = self.num_experts * torch.sum(importance * load)

        # 4. 分发与并行执行专家计算
        out_flat = torch.zeros_like(x_flat)

        # 按专家归类 Token 并执行计算 (掩码分发矩阵加速)
        for i, expert in enumerate(self.experts):
            # 找到哪些 Token 命中了当前专家 i
            mask = (topk_indices == i)
            if not mask.any():
                continue

            # 提取分配给该专家的 Token 索引及对应的 Top-K 权重系数
            token_indices, k_indices = torch.where(mask)
            expert_weights = topk_weights[token_indices, k_indices].unsqueeze(-1)

            # 专家前向计算，并加权回写到输出矩阵
            expert_out = expert(x_flat[token_indices])
            out_flat[token_indices] += expert_weights * expert_out

        return out_flat.view(B, T, C), aux_loss