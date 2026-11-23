# ==================== 【靶向精准隔离防火墙】置顶 ====================
import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# =========================================================================

from data.dataset import LMDBDataset
from data.tokenizer import CTCTokenizer

# ⚖️ 1. 实例化当前测试用的 98 码表
SECRET_95_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
tokenizer = CTCTokenizer(character=SECRET_95_CHARACTERS)

# 🚀 【精准路径修复】直接锚定到含有 data.mdb 的子数据集上
# 如果你的子文件夹名不叫 MJ/MJ_train，请改成你 data_lmdb_release/training 下真实的文件夹名
train_sub_dir = os.path.join(PROJECT_ROOT, "data/data_lmdb_release/training/MJ/MJ_train")

# 防御性回退：如果上面猜错了，就去 training 目录下随便抓一个存在的子文件夹
if not os.path.exists(train_sub_dir):
    base_train = os.path.join(PROJECT_ROOT, "data/data_lmdb_release/training")
    if os.path.exists(base_train):
        for root, dirs, files in os.walk(base_train):
            if "data.mdb" in files:
                train_sub_dir = root
                break

print(f"🎯 物理锁定目标训练子库路径: {train_sub_dir}")

try:
    train_dataset = LMDBDataset(root=train_sub_dir, tokenizer=tokenizer)

    print("\n🔬 [训练集 Tensor 强力破译观察]")
    print("=" * 60)

    for i in range(min(15, len(train_dataset))):
        sample = train_dataset[i]

        # 安全解包：适应 (img, label) 或 (img, label, ...) 各种格式
        if isinstance(sample, (tuple, list)):
            target_tensor = sample[1]
        else:
            target_tensor = sample

        # 核心解密逻辑
        if isinstance(target_tensor, torch.Tensor):
            try:
                decoded_text = tokenizer.decode(target_tensor.unsqueeze(0).cpu())[0]
            except Exception:
                try:
                    decoded_text = tokenizer.decode(target_tensor.cpu())
                except Exception:
                    decoded_text = "解码失败"
        else:
            decoded_text = str(target_tensor)

        # 打印结果
        raw_list = target_tensor.numpy().tolist() if isinstance(target_tensor, torch.Tensor) else list(target_tensor)
        print(f" 样本 {i + 1:<2} -> 原始数字: {raw_list}")
        print(f"          -> 98码表强解结果: {repr(decoded_text)}")
        print("-" * 60)

except Exception as e:
    print(f"❌ 读取发生错误: {e}")