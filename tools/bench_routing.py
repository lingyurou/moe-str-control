"""A3 支撑实验：dense / top1 / top2 三种路由的推理延迟与计算量。

两个口径都要报，缺一不可：
  1) token-专家对（token x expert pairs）—— 与硬件无关的计算量代理，写论文用这个；
  2) 实测墙钟延迟 —— 说明工程上真的快，但受 GPU 共享、Python 调度影响会有噪音。
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_decoder import GMoESemanticSTRModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_FF = 1024
N_EXP = 4


def build(mode):
    torch.manual_seed(0)
    m = GMoESemanticSTRModel(num_classes=98, d_model=256, nhead=8, num_layers=2,
                             num_experts=N_EXP, d_ff=D_FF, routing=mode,
                             backbone_type="svtr", img_h=32, img_w=100).to(DEV)
    return m.eval()


def count_pairs(model, B, L):
    """跑一次前向，统计所有层、所有 token 实际进入专家的总次数。"""
    imgs = torch.randn(B, 3, 32, 100, device=DEV)
    ys = torch.randint(0, 95, (B, L), device=DEV)
    total = 0
    with torch.no_grad():
        model(imgs, ys)
    for layer in model.layers:
        tpe = getattr(layer, "last_tokens_per_expert", None)
        if tpe is None:                       # dense：每个专家都吃全部 token
            total += B * L * N_EXP
        else:
            total += int(tpe.sum())
    return total


def latency(model, B, L, iters):
    imgs = torch.randn(B, 3, 32, 100, device=DEV)
    ys = torch.randint(0, 95, (B, L), device=DEV)
    with torch.no_grad():
        for _ in range(3):                    # warmup
            model(imgs, ys)
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(imgs, ys)
            if DEV == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    print("=" * 78)
    print("A3 支撑实验：路由模式的推理延迟与计算量（单卡 V100，与训练任务共享 GPU）")
    print("=" * 78)

    models = {m: build(m) for m in ("dense", "top1", "top2")}
    n_par = sum(p.numel() for p in models["dense"].parameters())
    print(f"参数量（三者相同）: {n_par/1e6:.3f} M | 专家数 {N_EXP} x d_ff {D_FF}\n")

    for B in (32, 128):
        L = 26
        base_pairs = None
        print(f"[batch={B}, 序列长={L}]")
        print(f"  {'routing':<8} {'专家前向次数':>12} {'相对计算量':>10} {'延迟(中位)':>12} {'相对延迟':>10}")
        rows = []
        for mode in ("dense", "top1", "top2"):
            pairs = count_pairs(models[mode], B, L)
            if base_pairs is None:
                base_pairs = pairs
            lat = latency(models[mode], B, L, iters=30 if B <= 32 else 12)
            rows.append((mode, pairs, pairs / base_pairs, lat))
        base_lat = rows[0][3]
        for mode, pairs, rel, lat in rows:
            print(f"  {mode:<8} {pairs:>12} {rel:>9.0%} {lat:>10.1f}ms {lat/base_lat:>9.2f}x")
        print()

    print("说明：'专家前向次数' 为两层 decoder、一个 batch 内的累计值；")
    print("      dense = B*L*4*2层，top1 = B*L*1*2层，top2 = B*L*2*2层。")


if __name__ == "__main__":
    main()
