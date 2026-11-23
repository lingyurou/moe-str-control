"""诊断 2：在 IIIT5k 全量上分离「解码链路损失」与「模型能力上限」。

三个口径：
  A. teacher forcing  —— 每个位置都喂正确历史，取 argmax，算 token 级与句级准确率。
                         这是模型在固定输入下的上界：完全不含误差累积。
  B. greedy 自回归     —— 从 <BOS> 起自己生成，遇 <EOS> 停（上限 25 步），与 test.py 同口径。
  C. beam search w=5  —— 同上但保留 5 条候选，按长度归一化选最优。

判读：
  A 远高于 B  → 暴露偏差 / 解码策略问题，链路有可捞的分。
  A ≈ B ≈ 89.9 → 模型泛化上限已定，与解码无关，必须动骨干/预训练/数据。

同时输出预测长度与 GT 长度的分布、编辑距离分布、错误样本样例。
"""

# Repository root, derived from this file so the tools run from any cwd.
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import collections
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, ROOT)

from data.dataset import LMDBDataset, AlignCollate              # noqa: E402
from data.tokenizer import CTCTokenizer                        # noqa: E402
from models.moe_decoder import GMoESemanticSTRModel, infer_config_from_state_dict  # noqa: E402

CHARSET_95 = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
TEST_ROOT = ROOT + "/data/data_lmdb_release/evaluation"


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_model(ckpt, device):
    sd = torch.load(ckpt, map_location=device)
    sd = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    cfg = infer_config_from_state_dict(sd)
    tok = CTCTokenizer(character=CHARSET_95)
    model = GMoESemanticSTRModel(
        num_classes=len(tok.full_vocab),
        d_model=cfg.get("d_model", 256),
        nhead=8,
        num_layers=cfg.get("num_layers", 2),
        num_experts=cfg.get("num_experts", 4),
        d_ff=cfg.get("d_ff", 1024),
        backbone_type=cfg.get("backbone_type", "svtr"),
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} cfg={cfg}")
    model.eval()
    return model, tok


@torch.no_grad()
def teacher_forcing_pass(model, loader, tok, device):
    """给正确历史，逐位置 argmax。返回 token 正确率与整句全对率。"""
    n_tok = n_tok_correct = n_seq = n_seq_correct = 0
    per_pos = collections.Counter()
    per_pos_tot = collections.Counter()
    for images, tgt_in, tgt_out, lengths in loader:
        images, tgt_in, tgt_out = images.to(device), tgt_in.to(device), tgt_out.to(device)
        logits, _, _ = model(images, tgt_in)
        pred = logits.argmax(-1)                       # [B, L]
        valid = tgt_out != tok.pad_id                  # 只算非 PAD 位置
        hit = (pred == tgt_out) & valid
        n_tok += int(valid.sum())
        n_tok_correct += int(hit.sum())
        for b in range(pred.size(0)):
            L = int(valid[b].sum())
            n_seq += 1
            if bool(hit[b, :L].all()):
                n_seq_correct += 1
            for p in range(L):
                per_pos_tot[p] += 1
                per_pos[p] += int(hit[b, p])
        del logits, pred
    return n_tok_correct / max(n_tok, 1), n_seq_correct / max(n_seq, 1), per_pos, per_pos_tot


@torch.no_grad()
def greedy_pass(model, loader, tok, device, max_len=25):
    """自回归贪心，遇 EOS 立即停（记录每个样本真实用到的步数）。"""
    preds, gts, steps = [], [], []
    for images, _tgt_in, tgt_out, lengths in loader:
        images = images.to(device)
        B = images.size(0)
        ys = torch.full((B, 1), tok.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        used = torch.zeros(B, dtype=torch.long, device=device)
        for t in range(max_len):
            logits, _, _ = model(images, ys)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            newly = (nxt.squeeze(1) == tok.eos_id) & (~finished)
            used[newly] = t + 1
            finished |= newly
        used[~finished] = max_len
        for row in ys:
            txt = []
            for idx in row[1:].tolist():
                if idx == tok.eos_id or idx == tok.pad_id:
                    break
                txt.append(tok.full_vocab[idx])
            preds.append("".join(txt))
        for row in tgt_out:
            txt = []
            for idx in row.tolist():
                if idx == tok.eos_id or idx == tok.pad_id:
                    break
                txt.append(tok.full_vocab[idx])
            gts.append("".join(txt))
        steps.extend(used.tolist())
        del logits
    return preds, gts, steps


@torch.no_grad()
def beam_pass(model, loader, tok, device, beam=5, max_len=25, alpha=0.6):
    """批内 beam search；每个样本独立维护 beam 条假设。"""
    preds, gts = [], []
    for images, _tgt_in, tgt_out, lengths in loader:
        images = images.to(device)
        B = images.size(0)
        # 假设张量：[B, beam, L]
        ys = torch.full((B, 1, 1), tok.bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(B, 1, device=device)
        done = [[] for _ in range(B)]      # 已完成（生成 EOS）的候选 (归一化分, 文本)

        for t in range(max_len):
            k = ys.size(1)
            flat = ys.reshape(B * k, -1)
            rep = images.repeat_interleave(k, dim=0)
            logits, _, _ = model(rep, flat)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)     # [B*k, V]
            logp = logp.view(B, k, -1)
            cand = scores.unsqueeze(-1) + logp                 # [B, k, V]
            vocab = cand.size(-1)
            flat_cand = cand.reshape(B, k * vocab)
            topv, topi = flat_cand.topk(min(beam, k * vocab), dim=-1)   # [B, beam]
            base = torch.div(topi, vocab, rounding_mode="floor")
            nxt = topi % vocab

            # 重建假设
            new_ys, new_scores = [], []
            for b in range(B):
                rows = []
                for j in range(topi.size(1)):
                    src = int(base[b, j])
                    tokid = int(nxt[b, j])
                    seq = torch.cat([ys[b, src], torch.tensor([tokid], device=device)])
                    rows.append(seq)
                    if tokid == tok.eos_id:
                        txt = [tok.full_vocab[i] for i in seq[1:-1].tolist()]
                        text = "".join(txt)
                        norm = float(topv[b, j]) / max(len(text), 1) ** alpha
                        done[b].append((norm, text))
                new_ys.append(torch.stack(rows))
                new_scores.append(topv[b])
            ys = torch.stack(new_ys)                            # [B, beam, L+1]
            scores = torch.stack(new_scores)                    # [B, beam]
            del logits, logp, cand, flat_cand

        for b in range(B):
            best = None
            if done[b]:
                best = max(done[b], key=lambda x: x[0])[1]
            if best is None:
                # 没有候选生成 EOS，取分数最高的未完成序列
                j = int(scores[b].argmax())
                txt = [tok.full_vocab[i] for i in ys[b, j, 1:].tolist()
                       if i not in (tok.eos_id, tok.pad_id)]
                best = "".join(txt)
            preds.append(best)
        for row in tgt_out:
            txt = []
            for idx in row.tolist():
                if idx == tok.eos_id or idx == tok.pad_id:
                    break
                txt.append(tok.full_vocab[idx])
            gts.append("".join(txt))
    return preds, gts


def report(tag, preds, gts, steps=None):
    n = len(gts)
    norm = lambda s: s.strip().lower()
    strict = sum(1 for p, g in zip(preds, gts) if p.strip() == g.strip())
    insens = sum(1 for p, g in zip(preds, gts) if norm(p) == norm(g))
    print(f"\n=== {tag} ===")
    print(f"样本 {n} | 句级严格 {strict/n*100:.2f}% | 句级大小写不敏感 {insens/n*100:.2f}%")
    if steps:
        c = collections.Counter(steps)
        print("生成步数分布(EOS位置):", dict(sorted(c.items())))
    d = collections.Counter()
    for p, g in zip(preds, gts):
        d[levenshtein(norm(p), norm(g))] += 1
    tot = sum(d.values())
    print("编辑距离分布:", {k: f"{v/tot*100:.1f}%" for k, v in sorted(d.items())[:6]})
    len_diff = collections.Counter(len(p) - len(g) for p, g in zip(preds, gts))
    print("预测长度-GT长度:", dict(sorted(len_diff.items())[:9]))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/a1_full/best.pth")
    ap.add_argument("--dataset", default="IIIT5k_3000")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-errors", type=int, default=15)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tok = build_model(args.checkpoint, device)
    tf = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ds = LMDBDataset(os.path.join(TEST_ROOT, args.dataset), tok, transform=tf)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=AlignCollate(tokenizer=tok))
    print(f"[data] {args.dataset} = {len(ds)} 张")

    tok_acc, seq_acc, per_pos, per_pos_tot = teacher_forcing_pass(model, loader, tok, device)
    print(f"\n=== A. 教师强制（不含误差累积，模型上界）===")
    print(f"token 级准确率 {tok_acc*100:.2f}% | 整句全对率 {seq_acc*100:.2f}%")
    print("逐位置 token 准确率(前12位):",
          " ".join(f"{p}:{per_pos[p]/per_pos_tot[p]*100:.1f}%"
                   for p in range(min(12, len(per_pos_tot)))))

    gp, gg, gsteps = greedy_pass(model, loader, tok, device)
    report("B. 贪心自回归（test.py 同口径）", gp, gg, gsteps)

    bp, bg = beam_pass(model, loader, tok, device, beam=args.beam)
    report(f"C. Beam search w={args.beam}", bp, bg)

    print(f"\n=== 错误样例（贪心，最多 {args.dump_errors} 条）===")
    shown = 0
    for p, g in zip(gp, gg):
        if p.strip().lower() != g.strip().lower() and shown < args.dump_errors:
            print(f"  GT={g!r:32} PRED={p!r}")
            shown += 1

    # 保存逐样本结果供后续分析
    out = f"logs/diag_decode_{args.dataset}.tsv"
    with open(out, "w") as f:
        f.write("gt\tgreedy\tbeam\n")
        for g, p, b in zip(gg, gp, bp):
            f.write(f"{g}\t{p}\t{b}\n")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
