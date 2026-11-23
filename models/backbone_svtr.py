"""SVTR 风格视觉骨干（可训练、可加载预训练权重）。

相对原始 3 层卷积骨干（约 0.37M 参数、单尺度、无残差）的改动：
  1. PatchEmbed：两层卷积下采样 stride 2，得到 16x50 特征图；
  2. LocalStage：2 个深度可分离卷积块，在 16x50 高分辨率上提取局部笔画特征（无注意力，便宜）；
  3. Merge：沿 H、W 各降采样 2 倍 -> 8x25 = 200 token，通道 64 -> 128；
  4. MixingStage：6 个 (dwconv + 多头自注意力 + MLP) 块，作用于 200 token；
  5. HeightMerging：(2,1) x3 把高度 8 -> 1，宽度保持 25，得到 T=25 的序列；
  6. LayerNorm + Linear 投影到 d_model。

为什么把全局注意力放在 200 token 而不是 800 token：
  注意力开销正比于 token 数的平方，800 个 token 时单卡吞吐只有 ~357 img/s
  （10 小时/ epoch），完全无法支撑 A1-A7 的实验量。本设计把高分辨率阶段
  交给便宜的深度卷积，注意力放到降采样后，吞吐提升约 3-4 倍，而宽度分辨率
  （决定能识别多长的字符串）仍保持 25 列不变。

设计参考 SVTR (IJCAI 2022) 的整体思路，但为控制算力做了结构调整，不是逐层复刻。

用法:
    bb = SVTRBackbone(d_model=256, pretrained=None)
    feat = bb(images)            # images [B,3,32,100] -> [B,25,d_model]
"""

import os

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LocalBlock(nn.Module):
    """纯局部块：深度卷积 + 逐点扩展，用于高分辨率阶段（无注意力）。"""

    def __init__(self, dim, drop=0.0, drop_path=0.0, k=3):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):                       # x: [B, C, H, W]
        x = x + self.drop_path(self.pw(self.act(self.dw(self.norm1(x)))))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MixingBlock(nn.Module):
    """局部（depthwise conv）+ 全局（self-attention）混合块。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, local_k=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.local = nn.Conv2d(dim, dim, local_k, 1, local_k // 2, groups=dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)
        self.gamma_local = nn.Parameter(torch.ones(dim))
        self.gamma_attn = nn.Parameter(torch.ones(dim))
        self.gamma_mlp = nn.Parameter(torch.ones(dim))

    def forward(self, x, hw):
        h, w = hw
        b, _n, c = x.shape

        # local mixing: 在 2D 网格上做 depthwise 卷积
        y = self.norm1(x).transpose(1, 2).reshape(b, c, h, w)
        y = self.local(y).flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.gamma_local * y)

        # global mixing: 降采样后的全图自注意力
        z = self.norm2(x)
        y, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop_path(self.gamma_attn * y)

        x = x + self.drop_path(self.gamma_mlp * self.mlp(self.norm3(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        half = max(dim // 2, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, half, 3, 2, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)                     # [B, dim, H/2, W/2]


class PatchMerging(nn.Module):
    """(kh, kw) 卷积下采样 + LayerNorm，输入输出均为序列 [B, N, C]。"""

    def __init__(self, dim, out_dim, kh=2, kw=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, out_dim, (kh, kw), (kh, kw), bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, hw):
        b, _n, c = x.shape
        feat = x.transpose(1, 2).reshape(b, c, *hw)
        feat = self.conv(feat).flatten(2).transpose(1, 2)
        return self.norm(feat)


class SVTRBackbone(nn.Module):
    def __init__(
        self,
        d_model=256,
        img_h=32,
        img_w=100,
        embed_dim=64,
        local_depths=2,
        depths=(6,),
        num_heads=(4,),
        mlp_ratio=4.0,
        drop=0.1,
        drop_path=0.05,
        pretrained=None,
    ):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        dim0 = embed_dim
        dim1 = embed_dim * 2

        self.patch_embed = PatchEmbed(3, dim0)
        h, w = img_h // 2, img_w // 2                     # 16, 50
        self.hw_hi = (h, w)

        # 高分辨率局部阶段（无注意力）
        self.local_stage = nn.ModuleList([
            LocalBlock(dim0, drop=drop, drop_path=drop_path * i / max(local_depths, 1))
            for i in range(local_depths)
        ])

        # 降采样到 8x25 = 200 token
        self.merge1 = PatchMerging(dim0, dim1, 2, 2)
        h, w = h // 2, w // 2                              # 8, 25
        self.hw_mid = (h, w)

        # 混合阶段（局部 + 全局）
        self.stage = nn.ModuleList([
            MixingBlock(dim1, num_heads[0], mlp_ratio, drop,
                        drop_path * (i + 1) / max(depths[0], 1))
            for i in range(depths[0])
        ])

        # 高度 8 -> 4 -> 2 -> 1，宽度保持
        self.merge2 = PatchMerging(dim1, dim1, 2, 1)
        self.merge3 = PatchMerging(dim1, dim1, 2, 1)
        self.merge4 = PatchMerging(dim1, dim1, 2, 1)

        self.out_norm = nn.LayerNorm(dim1)
        self.proj = nn.Linear(dim1, d_model)
        self.out_seqlen = w

        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, path):
        if not os.path.isfile(path):
            print(f"[backbone] 预训练权重不存在，跳过: {path}")
            return
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[backbone] 载入预训练 {path} | missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        feat = self.patch_embed(x)                        # [B, 64, 16, 50]
        for blk in self.local_stage:
            feat = blk(feat)

        x = feat.flatten(2).transpose(1, 2)               # [B, 800, 64]
        x = self.merge1(x, self.hw_hi)                    # [B, 200, 128]
        for blk in self.stage:
            x = blk(x, self.hw_mid)

        x = self.merge2(x, self.hw_mid)                   # [B, 100, 128]
        x = self.merge3(x, (self.hw_mid[0] // 2, self.hw_mid[1]))
        x = self.merge4(x, (self.hw_mid[0] // 4, self.hw_mid[1]))   # [B, 25, 128]
        return self.proj(self.out_norm(x))                # [B, 25, d_model]


def build_backbone(name="svtr", d_model=256, img_h=32, img_w=100, pretrained=None, **kw):
    name = (name or "svtr").lower()
    if name in ("svtr", "svtr_tiny", "svtr-lite"):
        return SVTRBackbone(d_model=d_model, img_h=img_h, img_w=img_w,
                            pretrained=pretrained, **kw)
    if name in ("legacy", "cnn", "toy"):
        return None  # 由调用方使用历史 3 层卷积实现，保持旧 checkpoint 兼容
    raise ValueError(f"未知 backbone: {name}")
