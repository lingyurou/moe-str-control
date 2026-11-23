#!/usr/bin/env python
"""G-MoE-STR 冒烟测试：验证本次改动的正确性。

覆盖 8 项检查：
  T1 码表 98 类、特殊 token 索引正确
  T2 三种配置的前向形状（svtr / legacy / 单专家稠密 FFN）
  T3 因果掩码：位置 t 的输出不受 t+1 之后 token 影响（最容易被改坏的地方）
  T4 路由：dense 权重和为 1；top1 恰为 one-hot；top2 至多 2 个非零
  T5 aux loss 落在 [1, N] 区间
  T6 参数量对齐：A1(4x1024) 与 A7-a(1x4096) 应基本相等
  T7 真实数据上 5 步训练，loss 必须下降（确认梯度链路通）
  T8 旧 checkpoint（legacy 骨干）仍能严格加载

运行:
    systemd-run --user -q --wait --pipe -p LimitAS=infinity \\
      bash -c 'PYTHONNOUSERSITE=1 python tools/smoke_test.py'
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset import AlignCollate, LMDBDataset          # noqa: E402
from data.str_aug import build_eval_transform                # noqa: E402
from data.tokenizer import CTCTokenizer                      # noqa: E402
from models.moe_decoder import GMoEDecoderLayer, GMoESemanticSTRModel  # noqa: E402
from train import CHARSET_95, EXPECTED_CLASSES               # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make_model(**kw):
    cfg = dict(num_classes=98, d_model=256, nhead=8, num_layers=2, num_experts=4,
               dropout=0.1, routing="dense", d_ff=1024,
               backbone_type="svtr", img_h=32, img_w=100)
    cfg.update(kw)
    return GMoESemanticSTRModel(**cfg).to(DEV)


def main():
    torch.manual_seed(0)

    print("\n[T1] 词表与特殊 token")
    tok = CTCTokenizer(character=CHARSET_95)
    check("len(full_vocab) == 98", len(tok.full_vocab) == EXPECTED_CLASSES,
          f"actual={len(tok.full_vocab)}")
    check("95 可见字符", len(tok.character) == 95, f"actual={len(tok.character)}")
    check("bos/eos/pad == 95/96/97", (tok.bos_id, tok.eos_id, tok.pad_id) == (95, 96, 97),
          f"actual={(tok.bos_id, tok.eos_id, tok.pad_id)}")

    print("\n[T2] 前向形状")
    imgs = torch.randn(4, 3, 32, 100, device=DEV)
    tgt = torch.randint(0, 98, (4, 12), device=DEV)
    out = make_model()(imgs, tgt)
    check("返回三元组", isinstance(out, (tuple, list)) and len(out) == 3, f"len={len(out)}")
    logits, gw, sel = out
    check("logits [4,12,98]", tuple(logits.shape) == (4, 12, 98), str(tuple(logits.shape)))
    check("gate_weights [2*4*12, 4]", tuple(gw.shape) == (2 * 4 * 12, 4), str(tuple(gw.shape)))
    check("selected [2*4*12, 1]", tuple(sel.shape) == (2 * 4 * 12, 1), str(tuple(sel.shape)))

    m_legacy = make_model(backbone_type="legacy")
    lg, gw2, _ = m_legacy(imgs, tgt)
    check("legacy 骨干输出一致形状", tuple(lg.shape) == (4, 12, 98), str(tuple(lg.shape)))
    check("legacy visual_seqlen == 26", m_legacy.visual_seqlen == 26)
    check("svtr visual_seqlen == 25", make_model().visual_seqlen == 25)

    print("\n[T3] 因果掩码（位置 t 不得看到 t+1 之后）")
    m = make_model(routing="dense").eval()
    tgt_a = torch.randint(0, 98, (3, 10), device=DEV)
    tgt_b = tgt_a.clone()
    tgt_b[:, 6:] = torch.randint(0, 98, tgt_b[:, 6:].shape, device=DEV)  # 只改后半段
    with torch.no_grad():
        la, _, _ = m(imgs[:3], tgt_a)
        lb, _, _ = m(imgs[:3], tgt_b)
    diff_front = (la[:, :6] - lb[:, :6]).abs().max().item()
    diff_back = (la[:, 6:] - lb[:, 6:]).abs().max().item()
    check("前 6 位不受后续 token 影响", diff_front < 1e-5, f"max|diff|={diff_front:.2e}")
    check("后 4 位确实改变（掩码非全屏蔽）", diff_back > 1e-4, f"max|diff|={diff_back:.2e}")

    print("\n[T4] 路由模式")
    layer = GMoEDecoderLayer(256, 8, 4, routing="dense").to(DEV).eval()
    x = torch.randn(2, 7, 256, device=DEV)
    mem = torch.randn(2, 25, 256, device=DEV)
    mask = torch.triu(torch.ones(7, 7, device=DEV), diagonal=1).bool()
    mask = mask.unsqueeze(0).repeat(2 * 8, 1, 1)
    with torch.no_grad():
        _, gw_d, sel_d = layer(x, mem, tgt_mask=mask)
    check("dense: 权重和为 1", torch.allclose(gw_d.sum(-1), torch.ones_like(gw_d.sum(-1)), atol=1e-5),
          f"max dev={(gw_d.sum(-1)-1).abs().max().item():.2e}")
    check("dense: 四项均为正", bool((gw_d > 0).all()))

    for mode, kmax in (("top1", 1), ("top2", 2)):
        l2 = GMoEDecoderLayer(256, 8, 4, routing=mode).to(DEV).eval()
        with torch.no_grad():
            l2(x, mem, tgt_mask=mask)
        mix = l2.last_mix_weights          # 实际参与加权的权重
        nonzero = (mix > 1e-8).sum(-1)
        check(f"{mode}: 非零项 <= {kmax}", int(nonzero.max()) <= kmax,
              f"max_nonzero={int(nonzero.max())}")
        check(f"{mode}: 权重和为 1", torch.allclose(mix.sum(-1), torch.ones(mix.shape[0], device=DEV), atol=1e-5))
        check(f"{mode}: 返回的概率仍是 softmax（供 aux loss 用）",
              bool((gw_t_of := l2.last_gate_probs).shape[-1] == 4 and (gw_t_of > 0).all()))

    print("\n[T5] aux loss 区间")
    m = make_model()
    logits, gw, _ = m(imgs, tgt)
    n = gw.shape[-1]
    p_i = torch.bincount(gw.argmax(-1), minlength=n).float() / gw.shape[0]
    aux = (n * torch.sum(p_i * gw.mean(0))).item()
    check(f"aux in [1, {n}]", 1.0 - 1e-3 <= aux <= n + 1e-3, f"aux={aux:.4f}")

    print("\n[T6] 参数量对齐（A1 vs A7-a）")
    n_a1 = sum(p.numel() for p in make_model(num_experts=4, d_ff=1024).parameters())
    n_a7a = sum(p.numel() for p in make_model(num_experts=1, d_ff=4096).parameters())
    n_a7b = sum(p.numel() for p in make_model(num_experts=1, d_ff=1024).parameters())
    n_old = sum(p.numel() for p in make_model(backbone_type="legacy").parameters())
    rel = abs(n_a1 - n_a7a) / n_a1
    print(f"       A1(4x1024)={n_a1/1e6:.3f}M  A7a(1x4096)={n_a7a/1e6:.3f}M  "
          f"A7b(1x1024)={n_a7b/1e6:.3f}M  legacy骨干={n_old/1e6:.3f}M")
    check("A1 与 A7-a 参数量误差 < 1%", rel < 0.01, f"rel={rel*100:.2f}%")

    print("\n[T7] 数据集递归扫描 + 真实数据 5 步训练")
    root = os.path.join(ROOT, "data/data_lmdb_release/training")
    if not os.path.isdir(root):
        check("训练数据存在", False, "路径缺失，跳过")
    else:
        from train import scan_lmdb_sources          # 延迟导入，避免循环
        ds_full, info = scan_lmdb_sources(root, tok, build_eval_transform(32, 100), True)
        names = [n for n, _ in info]
        print("       " + " | ".join(f"{n}: {c}" for n, c in info))
        check("递归扫描命中 MJ_train（原实现会漏掉）",
              any("MJ" in n for n in names), f"found={names}")
        check("递归扫描命中 ST", any("ST" in n for n in names), f"found={names}")
        check("已排除 test/valid 划分",
              not any(("test" in n.lower() or "valid" in n.lower()) for n in names),
              f"found={names}")
        check("总样本数 > 10M", len(ds_full) > 10_000_000, f"total={len(ds_full)}")

        sub = torch.utils.data.Subset(ds_full, range(0, 256))
        loader = torch.utils.data.DataLoader(
            sub, batch_size=32, shuffle=False,
            collate_fn=AlignCollate(tokenizer=tok), num_workers=2)
        m = make_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
        m.train()
        losses = []
        for i, (bi, tin, tout, _l) in enumerate(loader):
            if i >= 8:
                break
            bi, tin, tout = bi.to(DEV), tin.to(DEV), tout.to(DEV)
            opt.zero_grad(set_to_none=True)
            lo, gw, _ = m(bi, tin)
            ce = crit(lo.reshape(-1, lo.shape[-1]), tout.reshape(-1))
            ne = gw.shape[-1]
            pi = torch.bincount(gw.argmax(-1), minlength=ne).float() / gw.shape[0]
            aux = ne * torch.sum(pi * gw.mean(0))
            (ce + 0.01 * aux).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opt.step()
            losses.append(ce.item())
        first = sum(losses[:2]) / 2
        last = sum(losses[-2:]) / 2
        check("loss 下降", last < first, f"{first:.3f} -> {last:.3f}")
        check("无 NaN", all(x == x for x in losses), f"min={min(losses):.3f}")

    print("\n[T8] 旧 checkpoint 兼容")
    old = os.path.join(ROOT, "checkpoints/g_moe_semantic_epoch_30.pth")
    if os.path.isfile(old):
        m = make_model(backbone_type="legacy")
        sd = torch.load(old, map_location="cpu")
        try:
            m.load_state_dict(sd, strict=True)
            check("legacy 严格加载旧权重", True, f"{sum(v.numel() for v in sd.values())/1e6:.2f}M 参数")
        except RuntimeError as exc:
            check("legacy 严格加载旧权重", False, str(exc)[:120])
    else:
        print("       (旧 checkpoint 不存在，跳过)")

    print("\n" + "=" * 62)
    print(f"结果: {len(PASS)} 项通过, {len(FAIL)} 项失败")
    if FAIL:
        print("失败项: " + ", ".join(FAIL))
    print("=" * 62)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
