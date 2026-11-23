"""诊断 3：EOS 偏置扫描。

诊断 2 已确认贪心自回归与教师强制上界持平（无暴露偏差），但 47% 的错误样本
"预测比 GT 长"，即模型倾向于晚一步输出 <EOS>。这类偏差可以在解码端用
"给 EOS logit 加常数 delta" 直接纠正，无需重训。

本脚本在同一个数据集上扫多个 delta，报告句级准确率与平均预测长度。
"""

# Repository root, derived from this file so the tools run from any cwd.
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/tools")

from data.dataset import LMDBDataset, AlignCollate              # noqa: E402
from data.tokenizer import CTCTokenizer                        # noqa: E402
from models.moe_decoder import GMoESemanticSTRModel, infer_config_from_state_dict  # noqa: E402

CHARSET_95 = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
TEST_ROOT = ROOT + "/data/data_lmdb_release/evaluation"


def build_model(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    sd = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    cfg = infer_config_from_state_dict(sd)
    tok = CTCTokenizer(character=CHARSET_95)
    m = GMoESemanticSTRModel(
        num_classes=len(tok.full_vocab), d_model=cfg.get("d_model", 256), nhead=8,
        num_layers=cfg.get("num_layers", 2), num_experts=cfg.get("num_experts", 4),
        d_ff=cfg.get("d_ff", 1024), backbone_type=cfg.get("backbone_type", "svtr"),
    ).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, tok


@torch.no_grad()
def greedy(model, loader, tok, device, deltas, max_len=25):
    """对每个 delta 维护独立的一组 ys，共享同一张图但需要分别前向。"""
    preds = {d: [] for d in deltas}
    gts = []
    for images, _ti, tgt_out, _l in loader:
        images = images.to(device)
        B = images.size(0)
        batch_pred = {d: [] for d in deltas}
        live = {d: torch.full((B, 1), tok.bos_id, dtype=torch.long, device=device) for d in deltas}
        done = {d: torch.zeros(B, dtype=torch.bool, device=device) for d in deltas}
        for d in deltas:
            for _t in range(max_len):
                logits, _, _ = model(images, live[d])
                nxt_logits = logits[:, -1, :].clone()
                if d != 0.0:
                    nxt_logits[:, tok.eos_id] += d
                nxt = nxt_logits.argmax(-1, keepdim=True)
                live[d] = torch.cat([live[d], nxt], dim=1)
                done[d] |= (nxt.squeeze(1) == tok.eos_id)
                del logits, nxt_logits
            for row in live[d]:
                txt = []
                for idx in row[1:].tolist():
                    if idx in (tok.eos_id, tok.pad_id):
                        break
                    txt.append(tok.full_vocab[idx])
                batch_pred[d].append("".join(txt))
        for d in deltas:
            preds[d].extend(batch_pred[d])
        for row in tgt_out:
            txt = []
            for idx in row.tolist():
                if idx in (tok.eos_id, tok.pad_id):
                    break
                txt.append(tok.full_vocab[idx])
            gts.append("".join(txt))
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/a1_full/best.pth")
    ap.add_argument("--dataset", default="IIIT5k_3000")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deltas", default="0,0.5,1,1.5,2")
    # 新增：允许指定逐样本 tsv 落盘路径，避免多次评测互相覆盖
    ap.add_argument("--out", default=None,
                    help="逐样本 tsv 输出路径（默认 logs/diag_eos_<dataset>.tsv）")
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]

    device = torch.device(args.device)
    model, tok = build_model(args.checkpoint, device)
    tf = transforms.Compose([
        transforms.Resize((32, 100)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = LMDBDataset(os.path.join(TEST_ROOT, args.dataset), tok, transform=tf)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=AlignCollate(tokenizer=tok))
    print(f"[data] {len(ds)} 张 | deltas={deltas}")

    preds, gts = greedy(model, loader, tok, device, deltas)
    n = len(gts)
    gt_len = sum(len(g.strip()) for g in gts) / n
    print(f"\nGT 平均长度 {gt_len:.2f}")
    print(f"{'delta':>6} {'准确率':>9} {'平均预测长度':>12} {'偏长错误':>9} {'偏短错误':>9}")
    print("-" * 52)
    best = None
    for d in deltas:
        p = preds[d]
        acc = sum(1 for a, b in zip(p, gts) if a.strip().lower() == b.strip().lower()) / n
        plen = sum(len(x.strip()) for x in p) / n
        longer = sum(1 for a, b in zip(p, gts)
                     if a.strip().lower() != b.strip().lower() and len(a) > len(b))
        shorter = sum(1 for a, b in zip(p, gts)
                      if a.strip().lower() != b.strip().lower() and len(a) < len(b))
        print(f"{d:>6.1f} {acc*100:>8.2f}% {plen:>12.2f} {longer:>9} {shorter:>9}")
        if best is None or acc > best[1]:
            best = (d, acc)
    print("-" * 52)
    print(f"最佳 delta={best[0]} 准确率 {best[1]*100:.2f}%")

    out_path = args.out or f"logs/diag_eos_{args.dataset}.tsv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("gt\t" + "\t".join(f"d{d}" for d in deltas) + "\n")
        for i, g in enumerate(gts):
            f.write(g + "\t" + "\t".join(preds[d][i] for d in deltas) + "\n")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
