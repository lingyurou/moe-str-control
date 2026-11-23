"""诊断 1：只读数据层，统计测试集标签的长度分布与非法字符。

不加载 torch 模型，只用 lmdb，可在无 GPU 时快速跑。
目的：确认 max_len=25 的硬编码解码上限是否会截断真实标签，
      以及标签中是否存在 95 码表之外的字符（会被 encode 成空格）。
"""

# Repository root, derived from this file so the tools run from any cwd.
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import os
import sys
import collections

import lmdb

sys.path.insert(0, ROOT)

CHARSET_95 = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
VOCAB = set(CHARSET_95)

TEST_ROOT = ROOT + "/data/data_lmdb_release/evaluation"
NAMES = ["CUTE80", "IC03_860", "IC03_867", "IC13_1015",
         "IC13_857", "IC15_1811", "IC15_2077", "IIIT5k_3000", "SVT", "SVTP"]


def scan(path):
    env = lmdb.open(path, max_readers=32, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n = int(txn.get("num-samples".encode()))
        lens, bad_chars, empty = [], collections.Counter(), 0
        labels = []
        for i in range(1, n + 1):
            raw = txn.get(f"label-{i:09d}".encode())
            lab = raw.decode("utf-8") if raw else ""
            if not lab:
                empty += 1
            # 训练时一律 lower()，评测端 LMDBDataset 默认 lowercase=True
            lab_l = lab.lower()
            labels.append(lab_l)
            lens.append(len(lab_l))
            for ch in lab_l:
                if ch not in VOCAB:
                    bad_chars[ch] += 1
    env.close()
    lens.sort()
    return n, lens, bad_chars, empty, labels


def pct(arr, q):
    if not arr:
        return 0
    k = int(round((len(arr) - 1) * q))
    return arr[k]


print(f"{'数据集':<12} {'样本':>6} {'长度均值':>8} {'P95':>5} {'最大':>5} "
      f"{'>24字符':>8} {'占比':>7} {'非法字符':>9} {'空标签':>6}")
print("-" * 92)
total_long = total_all = 0
for name in NAMES:
    p = os.path.join(TEST_ROOT, name)
    if not os.path.isdir(p):
        print(f"{name:<12} 路径不存在")
        continue
    n, lens, bad, empty, _ = scan(p)
    long_cnt = sum(1 for x in lens if x > 24)
    total_long += long_cnt
    total_all += n
    bad_str = "".join(f"{k!r}x{v}" for k, v in bad.most_common(4)) if bad else "无"
    print(f"{name:<12} {n:>6} {sum(lens)/n:>8.1f} {pct(lens,0.95):>5} {lens[-1]:>5} "
          f"{long_cnt:>8} {long_cnt/n*100:>6.1f}% {bad_str:>9} {empty:>6}")

print("-" * 92)
print(f"合计 {total_all} 张，长度 >24（会被 max_len=25 截断）的有 "
      f"{total_long} 张，占 {total_long/total_all*100:.2f}%")
