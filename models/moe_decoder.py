import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbone_svtr import build_backbone
except ImportError:  # 允许以脚本方式直接运行
    from models.backbone_svtr import build_backbone


class SemanticRouter(nn.Module):
    """
    【TIP 级核心】语义门控路由网络
    根据输入的隐藏状态，动态计算出当前 Token 应该分流给哪一个专家。
    """

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: [B * L, d_model]
        logits = self.gate(x)
        # 使用 Softmax 转化为平滑的概率分布，用于多专家特征加权熔炼
        gate_weights = F.softmax(logits, dim=-1)
        return gate_weights, logits


class ExpertLayer(nn.Module):
    """
    【TIP 级核心】独立语义专家内核
    """

    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class GMoEDecoderLayer(nn.Module):
    """
    【TIP 级核心】集成了严格因果路由的多专家 Transformer Decoder 单层架构
    """

    def __init__(self, d_model, nhead, num_experts, dropout=0.1, routing="dense", d_ff=1024):
        super().__init__()
        # 显式声明 batch_first=True
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 路由与专家团配置
        self.num_experts = num_experts
        # dense: 全专家软加权（默认，论文主设定）；top1/top2: 硬路由（A3 消融）
        self.routing = routing
        self.router = SemanticRouter(d_model, num_experts)
        self.experts = nn.ModuleList(
            [ExpertLayer(d_model, d_ff=d_ff, dropout=dropout) for _ in range(num_experts)]
        )

        # 层归一化与残差防线
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        tgt: 文本特征 [B, L, d_model]
        memory: 视觉 Backbone 提取的特征 [B, S, d_model]
        tgt_mask: 已经物理平铺扩展为 [B * nhead, L, L] 的布尔严格因果掩码
        """
        # 🚀 步骤 1：文本内部的自注意力计算 (挂载绝对刚性的多头布尔掩码，严防向右看！)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 🚀 步骤 2：视觉-文本交叉注意力交互 (视觉特征允许全量看，不需要因果掩码)
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 🚀 步骤 3：进入多专家语义门控分流系统
        B, L, D = tgt.shape
        tgt_flat = tgt.view(-1, D)  # 拉平为 [B * L, d_model] 统一送入路由器

        # 计算当前时间步所有 Token 对各个专家的路由选择权重以及原始得分
        gate_weights, gate_logits = self.router(tgt_flat)  # [B * L, num_experts]

        # 🌟 动态获取当前时间步每个 Token 概率最高的 Top-1 专家索引，用于测试集方差计算
        _, selected_experts = torch.topk(gate_logits, k=1, dim=-1)

        # 按路由模式决定混合权重：gate_weights 始终保留 softmax 概率（供 aux loss 使用），
        # 真正参与加权的权重单独记为 mix_weights。
        #
        # [关键] 两条路径的计算量完全不同，这是 A3 消融的全部意义所在：
        #   dense : 全部 N 个专家都对「全部 token」做前向 —— 计算量 = N x d_ff，
        #           与一个 d_ff*N 的稠密 FFN 数学等价（这就是 A7 能和 A1 打平的原因）。
        #   top1/2: 每个 token 只送进被选中的 k 个专家 —— 计算量 = k x d_ff，
        #           未选中的专家**完全不执行**，这才是真正的条件计算。
        moe_output = torch.zeros_like(tgt_flat)

        if self.routing == "dense":
            mix_weights = gate_weights
            for i, expert in enumerate(self.experts):
                expert_out = expert(tgt_flat)
                moe_output = moe_output + mix_weights[:, i:i + 1] * expert_out
            self.last_tokens_per_expert = None

        elif self.routing in ("top1", "top2"):
            k = 1 if self.routing == "top1" else 2
            topv, topi = torch.topk(gate_weights, k=k, dim=-1)
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            mix_weights = torch.zeros_like(gate_weights).scatter(1, topi, topv)

            # 稀疏分派：按专家聚合其被分配到的 token，只对这些 token 做前向。
            #
            # [性能] 不能用「对每个专家做 nonzero/any 判断」的写法 —— 每次 .any()/.nonzero()
            # 都强制一次 GPU->CPU 同步，一层里会累积十几次同步，把省下的算力全吃回去
            # （实测这样写比 dense 还慢 1.6x）。改法：一次排序 + 一次 bincount，
            # 只把 counts 拉到 CPU（单次同步），之后全部用整数 slicing。
            T = tgt_flat.size(0)
            expert_ids = topi.reshape(-1)                       # [T*k]
            weight_flat = topv.reshape(-1)                      # [T*k]
            token_ids = torch.arange(T, device=tgt_flat.device).repeat_interleave(k)

            order = torch.argsort(expert_ids, stable=True)
            tok_sorted = token_ids.index_select(0, order)
            w_sorted = weight_flat.index_select(0, order)

            counts = torch.bincount(expert_ids, minlength=len(self.experts))
            offsets = torch.cumsum(counts, 0) - counts
            counts_l = counts.cpu().tolist()                    # 唯一一次同步
            offsets_l = offsets.cpu().tolist()

            for i, expert in enumerate(self.experts):
                c = int(counts_l[i])
                if c == 0:
                    continue
                s = int(offsets_l[i])
                idx = tok_sorted[s:s + c]
                wv = w_sorted[s:s + c].unsqueeze(-1)
                out = expert(tgt_flat.index_select(0, idx))
                moe_output = moe_output.index_add(0, idx, out * wv)

            # 记录每个专家实际处理的「token-专家对」数，供 A3 报告真实稀疏度
            # top1 时总和 = B*L；top2 时总和 = 2*B*L（每个 token 访问 2 个专家）
            self.last_tokens_per_expert = counts.detach()

        else:
            raise ValueError(f"未知 routing 模式: {self.routing}")

        # 暴露本层实际参与加权的权重，供 A5 路由探针与单测校验使用（不参与反传图）
        self.last_mix_weights = mix_weights.detach()
        self.last_gate_probs = gate_weights.detach()
        self.last_moe_output = moe_output.detach()   # 供稀疏性等价性单测使用

        # 还原回标准序列形状，并执行最后一层残差固化
        tgt = self.norm3(tgt + self.dropout(moe_output.view(B, L, D)))

        return tgt, gate_weights, selected_experts


class GMoESemanticSTRModel(nn.Module):
    """
    【IEEE TIP 满血完全体】绝对无作弊漏洞的自回归语义多专家场景文本识别模型
    """

    def __init__(
        self,
        num_classes=98,
        d_model=256,
        nhead=8,
        num_layers=2,
        num_experts=4,
        dropout=0.1,
        routing="dense",
        d_ff=1024,
        backbone_type="svtr",
        backbone_pretrained=None,
        img_h=32,
        img_w=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_experts = num_experts
        self.routing = routing
        self.backbone_type = backbone_type

        # 1. 视觉主干网络
        #    svtr   : SVTR 风格两级混合骨干（默认，A1 起用）
        #    legacy : 历史 3 层卷积骨干，仅用于兼容 6 月训练的旧 checkpoint
        if backbone_type in ("legacy", "cnn", "toy"):
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 26))  # 压缩高度，保留 26 个视觉水平序列切片
            )
            self.visual_seqlen = 26
        else:
            self.backbone = build_backbone(
                backbone_type, d_model=d_model, img_h=img_h, img_w=img_w,
                pretrained=backbone_pretrained,
            )
            if self.backbone is None:
                raise ValueError(f"无法构建 backbone: {backbone_type}")
            self.visual_seqlen = self.backbone.out_seqlen

        # 2. 文本词向量嵌入层与位置编码器
        self.token_embed = nn.Embedding(num_classes, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # 预留 100 长度的绝对位置编码

        # 3. 纵向多层多专家解码堆栈
        self.layers = nn.ModuleList([
            GMoEDecoderLayer(d_model, nhead, num_experts, dropout=dropout,
                             routing=routing, d_ff=d_ff)
            for _ in range(num_layers)
        ])

        # 4. 终极自回归大一统 98 类分类头
        self.prediction_head = nn.Linear(d_model, num_classes)

    def forward(self, images, tgt_input):
        """
        images: [B, 3, 32, 100]
        tgt_input: [B, L]
        """
        B, L = tgt_input.shape
        device = tgt_input.device

        # 🚀【终极铁闸】：生成标准的 bool 类型的严格因果下三角矩阵
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        # 💡 绝杀机制：为了防止 PyTorch 多头注意力发生广播二义性或越界忽略漏洞，
        # 我们直接将其物理平铺扩展到 [B * nhead, L, L] 的底层刚性形状！
        causal_mask = causal_mask.unsqueeze(0).repeat(B * self.nhead, 1, 1)  # [B * 8, L, L]

        # 2. 提取视觉特征
        if self.backbone_type in ("legacy", "cnn", "toy"):
            visual_feat = self.backbone(images)                  # [B, d_model, 1, 26]
            visual_feat = visual_feat.squeeze(2).permute(0, 2, 1)  # [B, 26, d_model]
        else:
            visual_feat = self.backbone(images)                  # [B, T, d_model] (SVTR 骨干已输出序列)

        # 3. 文本嵌入并注入位置编码
        tgt_emb = self.token_embed(tgt_input) + self.pos_encoder[:, :L, :]  # [B, L, 256]

        # 4. 纵向穿透多层专家塔
        all_gate_weights = []
        all_selected_experts = []
        for layer in self.layers:
            # 🚀 喂入被强制拉平到多头的物理安全掩码
            tgt_emb, gate_weights, selected_experts = layer(tgt_emb, visual_feat, tgt_mask=causal_mask)
            all_gate_weights.append(gate_weights)
            all_selected_experts.append(selected_experts)

        # 5. 输出大一统分类概率分布
        logits = self.prediction_head(tgt_emb)  # [B, L, num_classes]

        # 收集聚合所有层、所有 Batch、所有时间步的专家权重，用于训练时计算负载均衡
        total_gate_weights = torch.cat(all_gate_weights, dim=0)  # [Layers * B * L, num_experts]

        # 收集聚合被选中的专家索引，用于评测脚本在不破坏原有模型导出的前提下，计算准确的平衡方差
        total_selected_experts = torch.cat(all_selected_experts, dim=0)  # [Layers * B * L, 1]

        # 🌟 满血改造：不仅返回 logits，同时把权重分布和离散专家选择完美抛出
        return logits, total_gate_weights, total_selected_experts


def infer_config_from_state_dict(state_dict):
    """从 checkpoint 反推结构配置，避免评测端把 backbone / 专家数写死。

    旧 checkpoint（6 月那批）用 3 层卷积骨干，键名形如 backbone.features.0.weight；
    新 checkpoint 用 SVTR 骨干，键名形如 backbone.patch_embed.proj.0.weight。
    评测脚本据此自动选择，无需人工指定。
    """
    keys = list(state_dict.keys())
    cfg = {}

    if any(k.startswith("backbone.patch_embed.") for k in keys):
        cfg["backbone_type"] = "svtr"
    elif any(k.startswith("backbone.") for k in keys):
        # 旧版骨干是 nn.Sequential，键形如 backbone.0.weight（无 backbone.features 前缀）
        cfg["backbone_type"] = "legacy"

    layer_ids = []
    for k in keys:
        if k.startswith("layers.") or ".layers." in k:
            try:
                layer_ids.append(int(k.split("layers.")[1].split(".")[0]))
            except (IndexError, ValueError):
                pass
    if layer_ids:
        cfg["num_layers"] = max(layer_ids) + 1

    exp_ids = []
    for k in keys:
        if ".experts." in k:
            try:
                exp_ids.append(int(k.split(".experts.")[1].split(".")[0]))
            except (IndexError, ValueError):
                pass
    if exp_ids:
        cfg["num_experts"] = max(exp_ids) + 1

    for k, v in state_dict.items():
        if k.endswith("experts.0.linear1.weight"):
            cfg["d_ff"] = int(v.shape[0])
            break
    for k, v in state_dict.items():
        if k.endswith("token_embed.weight"):
            cfg["num_classes"] = int(v.shape[0])
            cfg["d_model"] = int(v.shape[1])
            break
    return cfg