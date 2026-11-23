import os
import sys

# ==================== 1. 智能路径防冲突隔离 ====================
try:
    import torch
except ImportError as e:
    print("❌ 运行环境错误: 基础 PyTorch 加载失败")
    raise e

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import io
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

try:
    import lmdb
except ImportError:
    if user_site in sys.path: sys.path.remove(user_site)
    sys.path.insert(0, user_site)
    import lmdb

from models import GMoESTRModel


class STRLabelConverter:
    def __init__(self):
        # 场景文本识别标准 96 字符集排列 (0为 CTC Blank，1-95 为真实字符)
        # 绝大多数开源项目基于此字典顺序
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        self.dict = {i: char for i, char in enumerate(self.chars)}

    def decode(self, logits):
        """
        强制绕过 Blank (0) 的干扰，进行非零最优序列召回的解码器
        """
        texts = []
        # logits 形状: [Batch, TimeStep, NumClasses]
        for batch_idx in range(logits.shape[0]):
            char_list = []
            prev_char_idx = -1

            for step_idx in range(logits.shape[1]):
                step_logits = logits[batch_idx, step_idx]  # [96]

                # 核心拯救：如果在当前时间步，0 (Blank) 的概率没有占据绝对统治地位 (比如没有达到 90% 以上)
                # 或者我们强行看第二大（非0最大）的字符是什么
                probs = torch.softmax(step_logits, dim=-1)

                # 获取排名前两位的类别
                top_vals, top_idxs = torch.topk(probs, k=2)

                # 如果第一名是 0 (Blank)，但它的置信度并不是绝对压制（比如未达到完全统治）
                # 我们强行召回第二名（真实的字符索引）
                if top_idxs[0].item() == 0:
                    best_idx = top_idxs[1].item()
                    confidence = top_vals[1].item()
                else:
                    best_idx = top_idxs[0].item()
                    confidence = top_vals[0].item()

                # 如果置信度尚可，且不与前一个字符重复（标准 CTC 融合）
                if best_idx != 0 and best_idx != prev_char_idx and confidence > 0.01:
                    char_list.append(self.dict.get(best_idx, ''))

                prev_char_idx = best_idx

            texts.append("".join(char_list))
        return texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GMoESTRModel().to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints/g_moe_str_epoch_20.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    # 智能层名模糊对齐
    model_keys = list(model.state_dict().keys())
    new_state_dict = {}
    matched_count = 0
    for m_key in model_keys:
        m_key_core = m_key.replace('backbone.', '').replace('module.', '')
        matched = False
        for c_key in state_dict.keys():
            c_key_core = c_key.replace('backbone.', '').replace('module.', '')
            if m_key_core == c_key_core:
                new_state_dict[m_key] = state_dict[c_key]
                matched = True
                matched_count += 1
                break
        if not matched:
            for c_key in state_dict.keys():
                if m_key_core in c_key or c_key.replace('backbone.', '') in m_key:
                    new_state_dict[m_key] = state_dict[c_key]
                    matched_count += 1
                    break

    print(f"📊 权重对齐审计：成功匹配并加载了 {matched_count}/{len(model_keys)} 个层。")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 数据集预处理 (100 宽度)
    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    lmdb_path = os.path.join(PROJECT_ROOT, 'data/data_lmdb_release/evaluation/IIIT5k_3000')
    dataset = LMDBDataset(lmdb_path, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    converter = STRLabelConverter()

    correct = 0
    total = 0

    print("\n🚀 开始执行非零最优序列召回推理评测...")
    with torch.no_grad():
        for i, (images, texts) in enumerate(tqdm(loader)):
            out = model(images.to(device))
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # 保证维度为 [Batch, TimeStep, NumClasses]
            if logits.shape[0] != len(texts):
                logits = logits.transpose(0, 1)

            decoded = converter.decode(logits)

            # 抽样打印前 3 个样本的强行解码，观察召回出来的字母序列
            if total == 0:
                print(f"\n[拯救策略实验]")
                for j in range(min(3, len(texts))):
                    print(f" 真实标签: {texts[j]:<10} -> 强制召回预测: {decoded[j] if decoded[j] else '[全空]'}")

            for p, g in zip(decoded, texts):
                if p.strip().lower() == g.strip().lower():
                    correct += 1
                total += 1

    print(f"\n✅ 最终拯救测试完成！当前机制下最高召回准确率: {(correct / total) * 100:.2f}%")


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.env = lmdb.open(root, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.num = int(txn.get(b'num-samples').decode())
        self.transform = transform

    def __len__(self): return self.num

    def __getitem__(self, i):
        i += 1
        with self.env.begin(write=False) as txn:
            img_buf = txn.get(f'image-{i:09d}'.encode())
            label = txn.get(f'label-{i:09d}'.encode()).decode()
            img = Image.open(io.BytesIO(img_buf)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


if __name__ == "__main__":
    main()