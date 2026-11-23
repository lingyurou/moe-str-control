#!/usr/bin/env python
"""G-MoE-STR 训练脚本（A1 版）。

相对原始 train.py 的修复与增强：
  1. [BUGFIX] forward 返回 3 个值 (logits, gate_weights, selected_experts)，
     原代码按 2 个解包，运行时必然 ValueError: too many values to unpack。
  2. [BUGFIX] 位置编码只有 100 长度，标签超长时会静默广播出错；现改为显式截断 + 断言。
  3. [新增] 全局随机种子固定（python / numpy / torch / cudnn），保证 3 seed 可复现。
  4. [新增] 可切换骨干：svtr（默认，A1 主设定）/ legacy（兼容旧 checkpoint）。
  5. [新增] 路由模式 dense / top1 / top2，对应 A3 消融。
  6. [新增] 训练期数据增强（透视、仿射、颜色、模糊），见 data/str_aug.py。
  7. [新增] AMP 混合精度、梯度裁剪、断点续训、best 权重保存。
  8. [新增] 训练中快速验证（IIIT5k 子集贪心解码），用于监控 A1 的 >=90% 决策规则。
  9. [新增] 码表断言：num_classes 必须等于 98，防止 96/98 口径错配重演。

用法:
    python train.py --config configs/a1_svtr.yaml
    python train.py --config configs/a1_svtr.yaml --dry-run
"""

import argparse
import json
import os
import random
import sys
import time

# [内存] 限制 glibc malloc arena 数量。默认每核最多 8 个 arena（32 核 → 256 个），
# DataLoader worker 里反复解码/增强图片时会造成严重碎片化，是 worker 内存随 epoch
# 膨胀的另一个来源。必须在 torch 导入和 worker 进程 fork 之前设置才生效。
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import AlignCollate, LMDBDataset            # noqa: E402
from data.str_aug import build_eval_transform, build_train_transform  # noqa: E402
from data.tokenizer import CTCTokenizer                        # noqa: E402
from models.moe_decoder import GMoESemanticSTRModel            # noqa: E402

CHARSET_95 = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
EXPECTED_CLASSES = 98   # 95 可见字符 + <BOS> + <EOS> + <PAD>


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_tokenizer():
    tok = CTCTokenizer(character=CHARSET_95)
    n = len(tok.full_vocab)
    if n != EXPECTED_CLASSES:
        raise AssertionError(
            f"码表口径错误：len(full_vocab)={n}，应为 {EXPECTED_CLASSES}。"
            f" 检查 CHARSET_95 是否恰好 {EXPECTED_CLASSES - 3} 个字符。"
        )
    return tok


def scan_lmdb_sources(root, tokenizer, transform, lowercase, max_depth=3,
                      skip_splits=("test", "valid", "val")):
    """递归扫描 root 下所有含 data.mdb 的目录（MJ/MJ_train 这类三层结构）。

    原实现只扫一层，而 MJ 的实际结构是 MJ/MJ_train/data.mdb，
    导致 lmdb.open 失败后被 try/except 静默跳过，27G 的 MJSynth 从未参与训练。
    这里改为递归，并按名称排除 test/valid 划分，避免训练集泄漏。
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"训练集目录不存在: {root}")

    found = []
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
            continue
        if "data.mdb" in files:
            name = os.path.basename(cur).lower()
            if any(s in name for s in skip_splits):
                print(f"[SKIP] 排除非训练划分: {cur}")
                dirs[:] = []
                continue
            found.append(cur)
            dirs[:] = []      # 已命中 LMDB，不再深入

    if not found:
        raise RuntimeError(f"{root} 下未找到任何有效 LMDB 数据集")

    parts, info = [], []
    for d in sorted(found):
        try:
            ds = LMDBDataset(d, tokenizer, transform, lowercase=lowercase)
        except Exception as exc:                       # noqa: BLE001
            print(f"[WARN] 跳过无法读取的子集 {d}: {exc}")
            continue
        if len(ds) > 0:
            parts.append(ds)
            info.append((os.path.relpath(d, root), len(ds)))

    if not parts:
        raise RuntimeError(f"{root} 下的 LMDB 均无法读取")
    return (parts[0] if len(parts) == 1 else ConcatDataset(parts)), info


# --------------------------------------------------------------------------- #
# 训练 / 验证
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    alpha=0.01, amp=True, scaler=None, log_every=100, max_steps=0):
    model.train()
    tot_loss = tot_ce = tot_aux = 0.0
    n_batches = 0

    for step, (images, tgt_input, tgt_output, _lengths) in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        tgt_input = tgt_input.to(device, non_blocking=True)
        tgt_output = tgt_output.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp):
            # [BUGFIX] 模型返回三元组，必须按三个变量解包
            logits, gate_weights, _selected = model(images, tgt_input)

            ce_loss = criterion(logits.reshape(-1, logits.shape[-1]),
                                tgt_output.reshape(-1))

            num_experts = gate_weights.shape[-1]
            p_i = torch.bincount(gate_weights.argmax(dim=-1),
                                 minlength=num_experts).float() / gate_weights.shape[0]
            f_i = gate_weights.mean(dim=0)
            aux_loss = num_experts * torch.sum(p_i * f_i)

            loss = ce_loss + alpha * aux_loss

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        tot_loss += loss.item()
        tot_ce += ce_loss.item()
        tot_aux += aux_loss.item()
        n_batches += 1

        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{time.strftime('%X')}] ep{epoch} [{step:>5}/{len(loader):>5}] "
                  f"loss {loss.item():.4f} (ce {ce_loss.item():.4f}, aux {aux_loss.item():.4f}) "
                  f"lr {lr:.2e}")

    n_batches = max(n_batches, 1)
    return tot_loss / n_batches, tot_ce / n_batches, tot_aux / n_batches, n_batches


@torch.no_grad()
def greedy_decode(model, images, bos_id, eos_id, pad_id, max_len=25):
    """无 KV cache 的贪心自回归解码，仅用于训练期监控。"""
    model.eval()
    b = images.shape[0]
    ys = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
    finished = torch.zeros(b, dtype=torch.bool, device=images.device)

    for _ in range(max_len):
        logits, _, _ = model(images, ys)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        finished |= nxt.squeeze(1).eq(eos_id)
        if bool(finished.all()):
            break
    return ys


@torch.no_grad()
def quick_val(model, loader, tokenizer, device, max_batches=6, amp=True):
    """在验证子集上算词准确率（标签已按 lowercase 协议统一）。"""
    model.eval()
    correct = total = 0
    for i, (images, _tgt_input, tgt_output, _l) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        ys = greedy_decode(model, images, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)
        preds = tokenizer.decode(ys[:, 1:])
        gts = tokenizer.decode(tgt_output)
        for p, gt in zip(preds, gts):
            total += 1
            correct += int(p == gt)
    return correct / max(total, 1), total


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_config(path):
    cfg = {}
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("_config_path", os.path.abspath(path))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/a1_svtr.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--backbone", default=None, choices=["svtr", "legacy"])
    ap.add_argument("--backbone-weights", default=None)
    ap.add_argument("--num-experts", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None,
                    help="每个专家的隐层宽度；A7 同参量对照用 num_experts=1 & d_ff=4096")
    ap.add_argument("--routing", default=None, choices=["dense", "top1", "top2"])
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--aug", default=None, choices=["none", "weak", "strong"])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=6,
                    help="快速验证使用的 batch 数（每批 100 张）；0 表示跑满整个验证集，"
                         "得到无偏但更慢的准确率（IIIT5k 全量 3000 张 = 30 批）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只跑几个 batch，用于冒烟验证")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="只用前 N 个训练样本（调试/快速验证可学性用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg.get("train", {})
    m = cfg.get("model", {})
    d = cfg.get("data", {})

    def pick(cli, key, default):
        return cli if cli is not None else g.get(key, default)

    seed = pick(args.seed, "seed", 42)
    epochs = pick(args.epochs, "epochs", 30)
    batch_size = pick(args.batch_size, "batch_size", 192)
    lr = pick(args.lr, "lr", 5e-4)
    workers = pick(args.workers, "workers", 8)
    alpha = args.alpha if args.alpha is not None else g.get("alpha", 0.01)
    amp = (not args.no_amp) and bool(g.get("amp", True))
    aug_level = args.aug or d.get("aug", "strong")
    val_every = pick(args.val_every, "val_every", 2)

    backbone = args.backbone or m.get("backbone", "svtr")
    routing = args.routing or m.get("routing", "dense")
    num_experts = args.num_experts or m.get("num_experts", 4)
    d_ff = args.d_ff or m.get("d_ff", 1024)
    d_model = m.get("d_model", 256)
    nhead = m.get("nhead", 8)
    num_layers = m.get("num_layers", 2)
    dropout = m.get("dropout", 0.1)
    img_h, img_w = d.get("img_h", 32), d.get("img_w", 100)

    data_root = args.data_root or d.get("train_root",
                                        "data/data_lmdb_release/training")
    eval_root = d.get("val_root", "data/data_lmdb_release/evaluation/IIIT5k_3000")
    out_dir = args.out_dir or g.get("out_dir", "checkpoints/a1_svtr")

    set_seed(seed, deterministic=bool(g.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = build_tokenizer()
    num_classes = len(tokenizer.full_vocab)
    print(f"[INFO] num_classes={num_classes} bos={tokenizer.bos_id} "
          f"eos={tokenizer.eos_id} pad={tokenizer.pad_id}")

    model = GMoESemanticSTRModel(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_experts=num_experts,
        dropout=dropout,
        routing=routing,
        d_ff=d_ff,
        backbone_type=backbone,
        backbone_pretrained=args.backbone_weights,
        img_h=img_h,
        img_w=img_w,
    ).to(device)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[INFO] backbone={backbone} routing={routing} experts={num_experts} d_ff={d_ff} "
          f"| 参数量 {n_par/1e6:.2f}M | visual_seqlen={model.visual_seqlen}")

    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] 续训 {args.resume} | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"       missing 前 5 项: {list(missing)[:5]}")

    train_tf = build_train_transform(img_h, img_w, aug_level)
    eval_tf = build_eval_transform(img_h, img_w)
    train_ds, info = scan_lmdb_sources(data_root, tokenizer, train_tf,
                                       lowercase=d.get("lowercase", True))
    for name, n in info:
        print(f"       - {name}: {n} 样本")
    if args.limit_train and args.limit_train < len(train_ds):
        train_ds = torch.utils.data.Subset(train_ds, range(args.limit_train))
        print(f"[INFO] 已限制训练样本数 -> {len(train_ds)}（调试模式）")
    print(f"[INFO] 训练样本总数 = {len(train_ds)} | 增强 = {aug_level}")

    # [A1v2] 可选的逐样本加权采样：修正训练集与测试集的分布错配
    # （纯数字 1.67% -> ~9%，长标签 2.6% -> ~7%）。weights 由
    # tools/build_sample_weights.py 生成，顺序与 scan_lmdb_sources 的
    # sorted(found) + ConcatDataset 顺序严格一致，加载时按样本数校验。
    sampler = None
    weights_path = d.get("sample_weights")
    if weights_path:
        if args.limit_train:
            raise SystemExit("[ABORT] sample_weights 与 --limit-train 不兼容（索引会错位）")
        w = np.load(weights_path)
        if len(w) != len(train_ds):
            raise SystemExit(f"[ABORT] 采样权重数 {len(w)} 与训练集样本数 {len(train_ds)} 不一致，"
                             f"请用 tools/build_sample_weights.py 重新生成")
        sampler = WeightedRandomSampler(
            torch.from_numpy(w).double(), num_samples=len(train_ds), replacement=True)
        print(f"[INFO] 启用加权采样: {weights_path} | 唯一样本 {len(w)} "
              f"| 加权总质量 {float(w.sum())/1e6:.1f}M（原始 {len(w)/1e6:.1f}M）")

    loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=workers,
        pin_memory=True, drop_last=True,
        # [关键] persistent_workers 必须为 False。
        # 实测泄漏：persistent worker 的内存随 epoch 线性增长（约 0.41 GB/worker/epoch），
        # A1 跑到 ep23 时 8 个 worker 各占 9.5 GB 匿名内存，叠加 LMDB 的 44 GB 文件映射
        # 直接把 123 GB 内存耗尽 —— 2026-11-01 01:57 A7 就是这么被 OOM killer 杀掉的。
        # 关掉后每轮重建 worker 池（启动开销数秒，相对 66000 步可忽略），内存按轮归零。
        persistent_workers=False,
        collate_fn=AlignCollate(tokenizer=tokenizer),
    )

    # 验证集分两类：
    #   select —— val_root，用于保存 best.pth（A1v2 起改用 MJ_valid 抽样，
    #             不再触碰 IIIT5k，避免「验证集=测试集」的选型偏差）
    #   monitor —— val_extra_roots，只打印趋势参与分析，不参与选型
    val_loader = None
    monitor_val = []                                   # [(name, loader), ...]
    if val_every > 0 and os.path.isdir(eval_root):
        val_ds = LMDBDataset(eval_root, tokenizer, eval_tf,
                             lowercase=d.get("lowercase", True))
        vms = int(d.get("val_max_samples", 0))
        tag = ""
        if vms and vms < len(val_ds):
            n_full = len(val_ds)
            gsel = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n_full, generator=gsel)[:vms].tolist()
            val_ds = torch.utils.data.Subset(val_ds, idx)
            tag = f"（seed={seed} 抽样 {vms}/{n_full}）"
        val_loader = DataLoader(val_ds, batch_size=100, shuffle=False,
                                num_workers=min(workers, 4), pin_memory=True,
                                collate_fn=AlignCollate(tokenizer=tokenizer))
        print(f"[INFO] 选型验证集 {eval_root}{tag}: {len(val_ds)} 样本")
    for extra_root in d.get("val_extra_roots", []) or []:
        if val_every > 0 and os.path.isdir(extra_root):
            eds = LMDBDataset(extra_root, tokenizer, eval_tf,
                              lowercase=d.get("lowercase", True))
            el = DataLoader(eds, batch_size=100, shuffle=False,
                            num_workers=min(workers, 4), pin_memory=True,
                            collate_fn=AlignCollate(tokenizer=tokenizer))
            monitor_val.append((os.path.basename(extra_root.rstrip("/")), el))
            print(f"[INFO] 监控验证集 {extra_root}: {len(eds)} 样本（不参与选型）")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=g.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_acc = [], 0.0
    max_steps = 5 if args.dry_run else 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss, avg_ce, avg_aux, _nb = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch,
            alpha=alpha, amp=use_amp, scaler=scaler, max_steps=max_steps,
        )
        scheduler.step()
        dt = time.time() - t0
        print(f"[{time.strftime('%X')}] epoch {epoch} | loss {avg_loss:.4f} "
              f"(ce {avg_ce:.4f}, aux {avg_aux:.4f}) | {dt:.0f}s "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        rec = {"epoch": epoch, "loss": avg_loss, "ce": avg_ce, "aux": avg_aux,
               "sec": round(dt, 1)}

        if val_loader is not None and epoch % val_every == 0:
            vb = args.val_batches if args.val_batches > 0 else 10 ** 9
            acc, n = quick_val(model, val_loader, tokenizer, device, amp=use_amp,
                               max_batches=vb)
            rec.update({"val_acc": round(acc, 4), "val_n": n})
            print(f"           - 选型验证 (MJ_valid 抽样): {acc*100:.2f}% (n={n})")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
                print(f"           - 新最优，已保存 best.pth ({acc*100:.2f}%)")
            for mname, mloader in monitor_val:
                macc, mn = quick_val(model, mloader, tokenizer, device, amp=use_amp,
                                     max_batches=vb)
                rec[f"mon_{mname}"] = round(macc, 4)
                print(f"           - 监控 {mname}: {macc*100:.2f}% (n={mn}) [不参与选型]")

        history.append(rec)
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pth"))

        if args.dry_run:
            print("[DRY-RUN] 冒烟通过，退出")
            break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 最优选型验证准确率 {best_acc*100:.2f}%（MJ_valid 抽样，不可与 IIIT5k 数字直接对比）"
          f" | 输出目录 {out_dir}")
    print("[NOTE] IIIT5k 的监控数字见 history.json 的 mon_IIIT5k_3000 字段；"
          "最终论文数字以十基准全量评测（test.py）为准。")


if __name__ == "__main__":
    main()
