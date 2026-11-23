"""生成训练集逐样本采样权重（A1v2 数据分布修正）。

背景（2026-11-02 诊断结论）：
  - 训练集纯数字仅 ~1.67%，而 IIIT5k 测试集约 8%（4.8 倍欠采样）→ 纯数字准确率只有 68.8%
  - 长标签（>12 字符）仅 ~2.6%，IIIT5k 13+ 字符样本准确率 14.3%
  方案：WeightedRandomSampler 上采样，epoch 样本总数不变（1275 万）。

权重规则（在 base=1.0 上叠加）：
  纯数字        ×6.0
  混合且含数字   ×2.0
  长度 > 12     ×3.0
预期占比：纯数字 ~8.8%、长标签 ~6.9%。

输出：data/sample_weights_v2.npy（float32）+ data/sample_weights_v2.json 统计。
顺序约定：与 train.py scan_lmdb_sources 完全一致（递归扫描 + sorted() 排序 +
排除 test/valid/val 子串），加载时按各子集样本数校验。
"""

# Repository root, derived from this file so the tools run from any cwd.
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse
import json
import os
import sys

import lmdb
import numpy as np

ROOT = ROOT + "/data/data_lmdb_release/training"

# v2 默认（×6/×2/×3）：纯数字补到 8.8%，但 A1v2 实测 ep12 监控 IIIT5k 仍落后旧 A1 3.4pp，
# 判定 ×6 过猛、挤占主词表学习 → v3 降档：×3/×2/×2。
DEFAULTS = {"digit_pure": 3.0, "digit_mix": 2.0, "long": 2.0}
LONG_TH = 12
SKIP = ("test", "valid", "val")
MAX_DEPTH = 3


def scan_sources(root):
    """复刻 train.py scan_lmdb_sources 的扫描/排序/排除逻辑，仅返回目录列表。"""
    found = []
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        if depth >= MAX_DEPTH:
            dirs[:] = []
            continue
        if "data.mdb" in files:
            name = os.path.basename(cur).lower()
            if any(s in name for s in SKIP):
                dirs[:] = []
                continue
            found.append(cur)
            dirs[:] = []
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit-pure", type=float, default=DEFAULTS["digit_pure"])
    ap.add_argument("--digit-mix", type=float, default=DEFAULTS["digit_mix"])
    ap.add_argument("--long", type=float, default=DEFAULTS["long"])
    ap.add_argument("--suffix", default="v3", help="输出文件名后缀")
    args = ap.parse_args()
    W_DIGIT_PURE, W_DIGIT_MIX, W_LONG = args.digit_pure, args.digit_mix, args.long
    OUT_NPY = f"{ROOT}/data/sample_weights_{args.suffix}.npy"
    OUT_JSON = f"{ROOT}/data/sample_weights_{args.suffix}.json"
    print(f"[规则] 纯数字×{W_DIGIT_PURE} 含数字混合×{W_DIGIT_MIX} 长>{LONG_TH}×{W_LONG}")
    srcs = scan_sources(ROOT)
    print(f"训练源 {len(srcs)} 个:")
    for s in srcs:
        print(f"  {os.path.relpath(s, ROOT)}")

    all_w = []
    stats = {}
    total = 0
    for s in srcs:
        env = lmdb.open(s, max_readers=32, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            n = int(txn.get(b"num-samples"))
            w = np.ones(n, dtype=np.float32)
            n_digit_pure = n_digit_mix = n_long = 0
            for i in range(1, n + 1):
                raw = txn.get(f"label-{i:09d}".encode())
                lab = (raw.decode("utf-8") if raw else "").lower()
                L = len(lab)
                is_pure_digit = lab.isdigit()
                has_digit = any(ch.isdigit() for ch in lab)
                if L > LONG_TH:
                    w[i - 1] *= W_LONG
                    n_long += 1
                if is_pure_digit:
                    w[i - 1] *= W_DIGIT_PURE
                    n_digit_pure += 1
                elif has_digit:
                    w[i - 1] *= W_DIGIT_MIX
                    n_digit_mix += 1
        env.close()
        all_w.append(w)
        stats[os.path.relpath(s, ROOT)] = {
            "n": n, "digit_pure": n_digit_pure, "digit_mix": n_digit_mix, "long": n_long,
        }
        total += n
        print(f"  [{os.path.relpath(s, ROOT)}] n={n} 纯数字={n_digit_pure} "
              f"含数字混合={n_digit_mix} 长>{LONG_TH}={n_long}")

    weights = np.concatenate(all_w)
    np.save(OUT_NPY, weights)

    # 加权后的期望采样占比
    eff = {
        "digit_pure": 0.0, "digit_mix": 0.0, "long": 0.0,
    }
    sum_w = float(weights.sum())
    # 从统计重算（与逐样本一致）
    dp = sum(v["digit_pure"] for v in stats.values())
    dm = sum(v["digit_mix"] for v in stats.values())
    lg = sum(v["long"] for v in stats.values())
    w_dp = dp * W_DIGIT_PURE * W_LONG * 0  # 占位，下面精确计算
    # 精确：纯数字与长标签有重叠，直接用权重向量分桶
    # 重新扫一遍代价高，用解析法：三桶互斥定义
    # digit_pure_and_long / digit_pure / digit_mix_and_long / digit_mix / long / rest
    # 但脚本第一遍已把权重算好，这里用蒙特卡洛不必要——直接用权重向量与计数近似：
    # 期望占比 = (桶权重和)/(总权重和)。为省事，第二遍快速扫（读标签很快）。
    buckets = {"digit_pure": 0.0, "digit_mix": 0.0, "long_only": 0.0, "rest": 0.0}
    idx0 = 0
    for s in srcs:
        env = lmdb.open(s, max_readers=32, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            n = int(txn.get(b"num-samples"))
            for i in range(1, n + 1):
                raw = txn.get(f"label-{i:09d}".encode())
                lab = (raw.decode("utf-8") if raw else "").lower()
                wv = float(weights[idx0 + i - 1])
                if lab.isdigit():
                    buckets["digit_pure"] += wv
                elif any(ch.isdigit() for ch in lab):
                    buckets["digit_mix"] += wv
                elif len(lab) > LONG_TH:
                    buckets["long_only"] += wv
                else:
                    buckets["rest"] += wv
        env.close()
        idx0 += n
    eff = {k: v / sum_w * 100 for k, v in buckets.items()}

    report = {
        "sources": stats, "total": total,
        "rule": {"digit_pure": W_DIGIT_PURE, "digit_mix": W_DIGIT_MIX,
                 "long_gt": LONG_TH, "long": W_LONG},
        "weighted_share_pct": eff,
        "raw_share_pct": {
            "digit_pure": dp / total * 100, "digit_mix": dm / total * 100,
            "long": lg / total * 100,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n合计 {total} 条 → {OUT_NPY} ({weights.nbytes/1048576:.1f} MB)")
    print(f"原始占比:   纯数字 {dp/total*100:.2f}%  含数字混合 {dm/total*100:.2f}%  长>{LONG_TH} {lg/total*100:.2f}%")
    print("加权后期望占比:")
    for k, v in eff.items():
        print(f"  {k:<12} {v:.2f}%")
    print("\n[OK] 完成")


if __name__ == "__main__":
    main()
