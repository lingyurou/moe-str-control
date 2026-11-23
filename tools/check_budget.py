"""校验四个配置的「参数预算」与「专家计算预算」，用于论文 Table 的公平对比声明。

关键概念区分：
  - params   : 全部参数（MoE 会把所有专家都计入显存，即使 top-1 推理只用其一）
  - expert_flops: 每个 token 实际经过的专家前向成本（= 调用对数/ token 数 x d_ff x 层数）
  这两者分开报，才能说清"同一参数量下算力不同"或"同一算力下参数量不同"。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_decoder import GMoESemanticSTRModel  # noqa: E402

CFGS = [
    ("A1  MoE 4x1024 dense", 4, 1024, "dense"),
    ("A3a MoE 4x1024 top1", 4, 1024, "top1"),
    ("A3b MoE 4x1024 top2", 4, 1024, "top2"),
    ("A7a dense 1x4096", 1, 4096, "dense"),
    ("A7b dense 1x1024", 1, 1024, "dense"),
]

B, L = 8, 26


def build(ne, dff, routing):
    return GMoESemanticSTRModel(
        num_classes=98, d_model=256, nhead=8, num_layers=2,
        num_experts=ne, d_ff=dff, routing=routing, backbone_type="svtr",
    ).cuda()


def main():
    x = torch.randn(B, 3, 32, 100).cuda()
    tg = torch.randint(0, 95, (B, L)).cuda()

    rows = []
    for name, ne, dff, routing in CFGS:
        m = build(ne, dff, routing).eval()
        n_par = sum(q.numel() for q in m.parameters())
        n_dec = sum(q.numel() for q in m.layers.parameters())
        with torch.no_grad():
            m(x, tg)
        calls = 0
        for layer in m.layers:
            c = getattr(layer, "last_tokens_per_expert", None)
            if c is None:
                # dense 模式不记录稀疏统计（因为每个专家都看了全部 token）
                calls += ne * B * L
            else:
                calls += int(c.sum())
        # 每 token 的专家 FLOPs 代理：calls/(B*L) 个专家前向 × d_ff × 2 个乘加
        flops = calls / (B * L) * dff
        rows.append((name, n_par / 1e6, n_dec / 1e6, calls, flops))
        del m
        torch.cuda.empty_cache()

    base_flops = rows[0][4]
    base_par = rows[0][1]
    head = "%-22s %10s %10s %10s %12s %12s" % (
        "config", "params(M)", "decoder(M)", "expert_fwd", "expert_FLOPs", "vs A1")
    print(head)
    print("-" * len(head))
    for name, p, pd, calls, fl in rows:
        print("%-22s %10.3f %10.3f %10d %12.0f %11.0f%%" % (
            name, p, pd, calls, fl, 100.0 * fl / base_flops))

    print("\n[配对检查] 论文需要三组对照：")
    d = {r[0]: r for r in rows}
    a1, a3a = d["A1  MoE 4x1024 dense"], d["A3a MoE 4x1024 top1"]
    a7a, a7b = d["A7a dense 1x4096"], d["A7b dense 1x1024"]

    # 组 1：参数与算力双双对齐 —— 4 个专家软混合 vs 1 个等宽 FFN
    print("  [组1] A1 vs A7a  参数 %.3fM vs %.3fM (差 %.2f%%) | 专家算力 %.0f vs %.0f (%.2f 倍)"
          % (a1[1], a7a[1], 100 * abs(a1[1] - a7a[1]) / a1[1], a1[4], a7a[4], a1[4] / a7a[4]))
    print("        -> 同参数、同 FLOPs。唯一变量是「拆分+路由」本身，最干净的归因对照。")

    # 组 2：算力对齐、参数更多 —— 稀疏条件容量 vs 窄稠密 FFN
    print("  [组2] A3a vs A7b 专家算力 %.0f vs %.0f (%.2f 倍) | 参数 %.3fM vs %.3fM (%.2f 倍)"
          % (a3a[4], a7b[4], a3a[4] / a7b[4], a3a[1], a7b[1], a3a[1] / a7b[1]))
    print("        -> 同算力、参数 1.9 倍。检验「存得多但算得少」是否真的换成精度。")

    # 组 3：参数对齐、算力降到 1/4 —— 效率主张的唯一合法依据
    print("  [组3] A3a vs A1  参数相同 (%.3fM) | 专家算力降到 %.0f%%"
          % (a3a[1], 100.0 * a3a[4] / a1[4]))
    print("        -> 精度不掉才能主张「同参数量下专家算力降 4 倍」。")


if __name__ == "__main__":
    main()
