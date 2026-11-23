import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ==========================================
# 1. 动态监控核心：MoEMonitor（直接内置，免去引包烦恼）
# ==========================================
class MoEMonitor:
    def __init__(self, model):
        self.stats = {}
        # 遍历模型所有子模块，寻找含有 'moe' 或 'gate' 的层
        for name, layer in model.named_modules():
            if 'moe' in name.lower() or 'gate' in name.lower():
                # 确保该层有参数，过滤掉纯容器层
                if len(list(layer.parameters())) > 0:
                    self.stats[name] = {"total": 0, "usage": None}
                    layer.register_forward_hook(self.create_hook(name))

        if self.stats:
            print(f"\n✅ [MoEMonitor] 成功挂载侦听器，关联模块: {list(self.stats.keys())}")
        else:
            print("\n⚠️ [MoEMonitor] 未找到包含 'moe' 或 'gate' 的指定模块，请检查层名。")

    def create_hook(self, name):
        def hook(module, input, output):
            # 兼容性处理：根据 MoE 输出结构解包
            # 常见结构 1: (output_tensor, routing_indices)
            # 常见结构 2: 仅输出 tensor，此时需要从 input 或 路由网络中抓取
            try:
                if isinstance(output, (tuple, list)) and len(output) > 1:
                    indices = output[1]
                else:
                    # 如果你的 MoE 层只返回特征，尝试从模块自身的历史属性中抓取（需配合模型代码）
                    return

                if not isinstance(indices, torch.Tensor):
                    return

                if self.stats[name]["usage"] is None:
                    # 动态初始化专家数量计数器
                    num_experts = int(indices.max().item() + 1)
                    self.stats[name]["usage"] = torch.zeros(num_experts, device=indices.device)

                # 统计各专家命中频次
                with torch.no_grad():
                    # 展平所有 token/sequence 的路由选择
                    flat_indices = indices.view(-1)
                    for idx in flat_indices:
                        if idx >= 0:  # 过滤掉 padding 或 mask
                            self.stats[name]["usage"][int(idx)] += 1
                            self.stats[name]["total"] += 1
            except Exception:
                pass  # 保证 Hook 绝对安全，不干扰主训练流程

        return hook

    def report(self):
        for name, data in self.stats.items():
            total = data["total"]
            if total > 0:
                print(f"\n📊 [📊 MoE 专家负载实时审计 | 层: {name}]")
                for i, count in enumerate(data["usage"]):
                    print(f"  ├─ 专家 {i:02d}: {count / total:.2%}")
                print(f"  └─ 总路由决策次数: {int(total)}")


# ==========================================
# 2. 标准训练主流程
# ==========================================
def main():
    # [配置初始化] 这里的参数请根据你的真实脚本调整
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 64

    print("📦 正在初始化 G-MoE-STR 神经网络...")
    # model = GMoESTR(...) # 实例化你项目里的真实模型
    # train_loader = DataLoader(...) # 你的真实数据加载器

    # 📝 示例虚拟占位（运行实际代码时请替换为你原本的模型和数据）
    model = nn.Sequential(nn.Linear(512, 512))

    model = model.to(device)

    # 🔔 【核心插入点】: 实例化模型并推送到 GPU 后，立刻挂载监控
    moe_monitor = MoEMonitor(model)

    # 优化器与损失函数（维持你原本的设定）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n🔥 联动完成，正式进入物理训练战场...")
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()

        # 假设你的 train_loader 返回 (images, labels)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 损失计算（若有 Aux Loss，请加上你的计算公式）
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 🕒 实时流日志打印
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{time.strftime('%H:%M:%S')}] Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] | Loss: {loss.item():.4f} | Time/Batch: {elapsed / (batch_idx + 1):.2f}s")

            # 📊 【核心插入点】: 每 1000 个迭代步长，强制进行一次专家负载审计报告
            if batch_idx % 1000 == 0 and batch_idx > 0:
                moe_monitor.report()

        # Epoch 结束保存权重
        checkpoint_path = f"checkpoints/g_moe_semantic_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Epoch {epoch} 训练结束，权重成功固化至: {checkpoint_path}")


if __name__ == "__main__":
    main()