import os
import sys
import time

# Run from any working directory: put the repository root on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 完美拉齐训练端原生数据组件与大模型
from data.dataset import LMDBDataset, AlignCollate
from data.tokenizer import CTCTokenizer
from models.moe_decoder import GMoESemanticSTRModel, infer_config_from_state_dict


def decode_predictions(ys, alphabet):
    """
    将自回归生成的批量 Token ID 序列（形状：[B, L]）转化为干净的明文字符串列表
    """
    pred_texts = []
    for ids in ys:
        text = ""
        # 从第二个字符开始（跳过开头的 <BOS>）
        for idx in ids[1:]:
            char = alphabet[idx.item()]
            # 🚀 精准修复：匹配你分词器中真实的尖括号结束符 <EOS>
            if char == '<EOS>' or char == '[EOS]':
                break
            if char not in ['[BOS]', '[PAD]', '[UNK]', '<BOS>', '<PAD>', '<UNK>']:
                text += char
        pred_texts.append(text)
    return pred_texts


def get_academic_test_loaders(tokenizer, test_transform, align_collate):
    """
    🎯 【多源学术评测集自动流式挂载】
    """
    test_root = os.path.join(PROJECT_ROOT, "data", "data_lmdb_release", "evaluation")

    # 国际顶刊学术基准 10 大核心测试集
    dataset_names = [
        "CUTE80", "IC03_860", "IC03_867", "IC13_1015",
        "IC13_857", "IC15_1811", "IC15_2077", "IIIT5k_3000",
        "SVT", "SVTP"
    ]

    loaders = {}
    print(f"[{time.strftime('%X')}] 🔍 正在检索并物理绑定 10 大主流测试数据集...")

    for name in dataset_names:
        lmdb_path = os.path.join(test_root, name)
        if not os.path.exists(lmdb_path):
            print(f"      [WARNING] ⚠️ 路径不存在，已跳过评测集: [{name}] -> {lmdb_path}")
            continue

        try:
            test_dataset = LMDBDataset(root=lmdb_path, tokenizer=tokenizer, transform=test_transform)
            if len(test_dataset) > 0:
                loaders[name] = DataLoader(
                    test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=align_collate
                )
                print(f"      └─ 成功接入测试集: [{name:<11}] | 样本总数: {len(test_dataset)}")
        except Exception as e:
            print(f"      [ERROR] ❌ 挂载测试集 {name} 失败，原因: {e}")
            continue

    return loaders


def execute_real_evaluation(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{time.strftime('%X')}] 🚀 开始执行 98 码表绝对自回归学术级【全量实弹】评测...")

    # 1. ⚖️ 严格锁死标准 95+3 字符集体系
    SECRET_95_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    tokenizer = CTCTokenizer(character=SECRET_95_CHARACTERS)
    num_classes = len(tokenizer.full_vocab)  # 严格等于 98 类

    alphabet = tokenizer.full_vocab
    print(f"[INFO] 验证端自回归安全边界强行锁死 num_classes = {num_classes}")

    # 2. 🎯 先读权重，再从权重反推结构（不再把 backbone / 专家数写死）
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 未找到指定的权重固化文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                              dict) and 'model_state_dict' in checkpoint else checkpoint
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    inferred = infer_config_from_state_dict(clean_state_dict)
    print(f"[INFO] 从权重推断结构: {inferred}")

    if inferred.get("num_classes") and inferred["num_classes"] != num_classes:
        raise AssertionError(
            f"码表口径不匹配：权重为 {inferred['num_classes']} 类，当前分词器为 {num_classes} 类。"
            " 拒绝在不一致的码表上评测（这正是旧版评测失真的原因之一）。"
        )

    num_experts = inferred.get("num_experts", 4)
    model = GMoESemanticSTRModel(
        num_classes=num_classes,
        d_model=inferred.get("d_model", 256),
        nhead=8,
        num_layers=inferred.get("num_layers", 2),
        num_experts=num_experts,
        d_ff=inferred.get("d_ff", 1024),
        backbone_type=inferred.get("backbone_type", "svtr"),
    ).to(device)

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] 权重未完全对齐 | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"       missing 前 5: {list(missing)[:5]}")
    else:
        print("[INFO] 权重严格对齐（0 missing / 0 unexpected）")

    model.eval()
    print(f"[SUCCESS] 👑 真实 G-MoE-STR 专家网络自回归权重装载成功")
    print("=" * 115)

    # 4. 📦 接入真实测试管线组件
    test_transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    align_collate = AlignCollate(tokenizer=tokenizer)

    eval_loaders = get_academic_test_loaders(tokenizer, test_transform, align_collate)
    if len(eval_loaders) == 0:
        print("[ERROR] ❌ 未能接通任何有效的物理评测数据集，测试终止。")
        return

    # 5. 🎬 物理全实弹自回归推理大循环
    print("\n" + "=" * 50 + " 进入前线实弹评测战场 " + "=" * 50)

    # 🌟 新增：创建一个全量大字典，用来分别统计这 10 个数据集各自的专家分配情况
    dataset_moe_logs = {}

    with torch.no_grad():
        for d_name, loader in eval_loaders.items():
            strict_correct = 0
            insensitive_correct = 0
            total_samples = 0

            # 🌟 新增：用来记录当前这个数据集上所有 Token 的路由选择
            current_dataset_experts = []

            for images, _, target_ids, _ in tqdm(loader, desc=f"正在自回归物理攻坚数据集: {d_name:<11}"):
                images = images.to(device)
                B = images.size(0)
                total_samples += B

                # 🛡️ 绝对自回归单步贪婪解码防线
                max_len = 25
                ys = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=device)

                # 逐字推理步进
                for i in range(max_len):
                    # 🚀 注意：在上一轮对模型的 G_MoE_STR 代码修改后，
                    # 你的大模型前向传播现在除了返回 logits，还会传出 gate_logits 和选中的专家 selected_experts
                    # 如果你在 GMoESemanticSTRModel 的最外层包装里也向上传递了该变量，可以直接在此捕获：
                    out = model(images, ys)

                    # 智能兼容拦截：如果模型外层包装还没把专家索引传出来，我们安全解包
                    if isinstance(out, (tuple, list)) and len(out) >= 3:
                        logits, gate_logits, selected_experts = out[0], out[1], out[2]
                        # 记录当前时间步全量 Token 的专家投递索引
                        current_dataset_experts.append(selected_experts.cpu().view(-1))
                    else:
                        logits = out[0] if isinstance(out, (tuple, list)) else out

                    next_token_logits = logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    ys = torch.cat([ys, next_tokens], dim=1)

                # 还原干净的模型预测结果
                pred_texts = decode_predictions(ys, alphabet)

                # 真实标签 Tensor 洗回字符串列表
                gt_texts = tokenizer.decode(target_ids.cpu())

                # 📊 真实多维学术级双轨比对
                for pred_text, gt_text in zip(pred_texts, gt_texts):
                    pred_clean = pred_text.strip()
                    gt_clean = gt_text.strip()

                    # 标准 1：大小写敏感
                    if pred_clean == gt_clean:
                        strict_correct += 1
                    # 标准 2：大小写不敏感
                    if pred_clean.lower() == gt_clean.lower():
                        insensitive_correct += 1

            # 计算命中率
            strict_acc = (strict_correct / total_samples) * 100 if total_samples > 0 else 0
            insensitive_acc = (insensitive_correct / total_samples) * 100 if total_samples > 0 else 0

            print(
                f"\n统计测试集: [{d_name:<11}] | 真实总样本: {total_samples:<5} | 认对(严格): {strict_correct:<4} | 认对(顶刊): {insensitive_correct:<4}")
            print(f" ├─ ❌ 严格准确率 (Case-Sensitive):   {strict_acc:.2f}%")
            print(f" └─ 👑 顶刊标准准确率 (Case-Insensitive): {insensitive_acc:.2f}%")

            # ==================== 🌟 核心增补：单数据集 MoE 方差就地统计与打印 ====================
            if len(current_dataset_experts) > 0:
                all_tokens_experts = torch.cat(current_dataset_experts)
                # 统计每一个专家分配到的 Token 绝对总量
                expert_counts = torch.bincount(all_tokens_experts, minlength=num_experts).float()
                # 转化为相对占比
                expert_props = expert_counts / all_tokens_experts.size(0)
                # 计算方差 Variance
                dataset_variance = torch.var(expert_props).item()

                print(f" ├─ 📊 各专家分配 Token 计数: {expert_counts.tolist()}")
                print(f" ├─ 📊 各专家负载相对比例: {[f'{p * 100:.2f}%' for p in expert_props.tolist()]}")
                print(f" └─ 🎯 顶刊 Table 2 门控平衡方差 (L_aux Var): {dataset_variance:.6f}")

                # 缓存到全局词典中，供最后打印全局合并大报表使用
                dataset_moe_logs[d_name] = all_tokens_experts

            print("-" * 115)

    # ==================== 🌟 核心增补：全量全数据集多专家大合并报表输出 ====================
    if len(dataset_moe_logs) > 0:
        print("\n" + "=" * 45 + " 👑 全量数据集级 MoE 负载均衡总决算大表 " + "=" * 45)
        # column count follows the checkpoint: dense controls have a single expert,
        # so the table must not assume four of them.
        header = f"{'数据集名称 (Dataset)':<20} | " + " | ".join(
            f"{'专家%d占比' % (i + 1):<10}" for i in range(num_experts)
        ) + " | 🎯 顶刊指标 (L_aux Var)"
        rule_len = 24 + 13 * num_experts
        print(header)
        print("-" * rule_len)

        all_dataset_combined = []
        for name, tokens_log in dataset_moe_logs.items():
            counts = torch.bincount(tokens_log, minlength=num_experts).float()
            props = (counts / tokens_log.size(0)).tolist()
            variance = torch.var(counts / tokens_log.size(0)).item() if num_experts > 1 else 0.0
            all_dataset_combined.append(tokens_log)

            cells = " | ".join(f"{p * 100:8.2f}%" for p in props)
            print(f"{name:<20} | {cells} | {variance:.6f}")

        # 10 大数据集彻底融合成一个超大全局域的终极方差（直接塞进你的 Table 2 或摘要里）
        global_tokens = torch.cat(all_dataset_combined)
        g_counts = torch.bincount(global_tokens, minlength=num_experts).float()
        g_props = (g_counts / global_tokens.size(0)).tolist()
        g_variance = torch.var(g_counts / global_tokens.size(0)).item() if num_experts > 1 else 0.0

        print("-" * rule_len)
        g_cells = " | ".join(f"{p * 100:8.2f}%" for p in g_props)
        print(f"{'👑 全局总域大平均 (Global)':<20} | {g_cells} | \033[1;32m{g_variance:.6f}\033[0m")
        print("=" * rule_len + "\n")

    print(f"[{time.strftime('%X')}] 🎉 G-MoE-STR 全量实弹自回归评测长征全部圆满结束！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MoE-STR 学术级全功能实弹自回归评测端")
    parser.add_argument("--checkpoint", type=str, required=True, help="需要被测分的权重存储路径")
    args = parser.parse_args()

    execute_real_evaluation(args.checkpoint)