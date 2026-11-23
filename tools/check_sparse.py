"""A3 前置校验：稀疏分派（top1/top2）必须与稠密参考在数值上等价。

为什么必须验：
  真稀疏实现是「按专家聚合 token + index_select」，一旦索引或权重拼错，
  训练照样能跑、loss 照样会降，但结果是错的，且看不出来。
  所以在启动两天规模的 A3 训练前，必须证明：
    1) top-k 稀疏输出 == 用同一组路由权重的稠密加权参考（逐元素）；
    2) 梯度只回流到被选中的专家（未选中专家 grad 为 0）；
    3) 每个专家实际处理的 token 数与路由统计自洽。
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_decoder import GMoEDecoderLayer  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D, S = 8, 26, 256, 25
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def build(mode):
    torch.manual_seed(0)
    layer = GMoEDecoderLayer(D, 8, 4, dropout=0.0, routing=mode, d_ff=1024).to(DEV)
    return layer.eval()


def dense_reference(mode, tgt, mem, mask, seed=0):
    """用 dense 路径拿到各专家的完整输出，再按 top-k 权重手工加权，作为参考值。"""
    layer = build("dense")
    # 与 sparse 版本共享同一套参数，保证可比
    ref_layer = build(mode).eval()
    layer.load_state_dict(ref_layer.state_dict())

    captured = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            captured[i] = out.detach()
        return hook

    for i, e in enumerate(layer.experts):
        e.register_forward_hook(make_hook(i))

    with torch.no_grad():
        layer(tgt, mem, tgt_mask=mask)                      # 触发全部专家前向
        _ = ref_layer(tgt, mem, tgt_mask=mask)              # 稀疏实现，拿到 mix_weights
    w = ref_layer.layers_weights if False else ref_layer.last_mix_weights  # [T, N]
    flat = None
    return captured, w, ref_layer


def main():
    torch.manual_seed(0)
    tgt = torch.randn(B, L, D, device=DEV)
    mem = torch.randn(B, S, D, device=DEV)
    mask = torch.triu(torch.ones(L, L, device=DEV, dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).repeat(B * 8, 1, 1)

    print("=" * 70)
    print("A3 前置校验：稀疏分派等价性 / 梯度路由 / 稀疏度")
    print("=" * 70)

    for mode, k in (("top1", 1), ("top2", 2)):
        print(f"\n[{mode}]")
        layers = [GMoEDecoderLayer(D, 8, 4, dropout=0.0, routing=m) for m in ("dense", mode)]
        layers[0].to(DEV).eval()
        layers[1].to(DEV).eval()
        layers[0].load_state_dict(layers[1].state_dict())
        dense_layer, sparse_layer = layers

        # ---- 1. 数值等价 ----
        captured = {}
        for i, e in enumerate(dense_layer.experts):
            e.register_forward_hook(
                (lambda idx: (lambda _m, _inp, out: captured.__setitem__(idx, out.detach())))(i))

        with torch.no_grad():
            dense_layer(tgt, mem, tgt_mask=mask)
            sparse_layer(tgt, mem, tgt_mask=mask)

        w = sparse_layer.last_mix_weights                          # [B*L, N] top-k 归一化权重
        ref = torch.zeros(B * L, D, device=DEV)
        for i in range(4):
            ref = ref + w[:, i:i + 1] * captured[i]
        got = sparse_layer.last_moe_output
        maxdiff = (ref - got).abs().max().item()
        check(f"{mode}: 稀疏输出 == 稠密参考", maxdiff < 1e-5, f"max|diff|={maxdiff:.2e}")

        # ---- 2. 稀疏度 ----
        tpe = sparse_layer.last_tokens_per_expert
        total = int(tpe.sum())
        check(f"{mode}: 专家处理的 token 总数 == {k}x(B*L)",
              total == k * B * L, f"实际 {total} vs 期望 {k * B * L}")
        check(f"{mode}: 非零项个数 == {k}", int((w > 0).sum(-1).max()) == k)

        # ---- 3. 梯度只回流到被选中的专家 ----
        sparse_layer.zero_grad(set_to_none=True)
        tgt_g = tgt.clone().requires_grad_(True)
        out, _gw, _sel = sparse_layer(tgt_g, mem, tgt_mask=mask)
        out.sum().backward()
        grad_norms = []
        for e in sparse_layer.experts:
            g = sum((p.grad.abs().sum() if p.grad is not None else torch.zeros(())).item()
                    for p in e.parameters())
            grad_norms.append(g)
        # 反向验证：unused 专家（本 batch 未被分配）grad 必须精确为 0
        unused = [i for i in range(4) if int(tpe[i]) == 0]
        if unused:
            zero_ok = all(grad_norms[i] == 0.0 for i in unused)
            check(f"{mode}: 未分配专家梯度为 0", zero_ok,
                  f"unused={unused} grads={[round(grad_norms[i], 4) for i in unused]}")
        else:
            check(f"{mode}: 4 专家均被分配（恰好覆盖）", all(g > 0 for g in grad_norms),
                  f"grads={[f'{g:.2e}' for g in grad_norms]}")

        # ---- 4. 与 dense 的显存/计算差异（只报 token 数）----
        print(f"       每专家处理 token-专家对: {tpe.tolist()}  "
              f"(dense 时每个都是 {B * L}，合计 {4 * B * L})")

    print("\n" + "=" * 70)
    print("结论:", "全部通过，可以启动 A3" if ok_all else "存在失败项，禁止启动 A3")
    print("=" * 70)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
