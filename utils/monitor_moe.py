import torch


class MoEMonitor:
    def __init__(self, model):
        self.stats = {}
        # 挂载 Hook
        for name, layer in model.named_modules():
            if 'moe' in name.lower():
                self.stats[name] = {"total": 0, "usage": None}
                layer.register_forward_hook(self.create_hook(name))
                print(f"✅ 监控已挂载至: {name}")

    def create_hook(self, name):
        def hook(module, input, output):
            # 假设输出格式为 (features, indices)
            _, indices = output
            if self.stats[name]["usage"] is None:
                num_experts = indices.max().item() + 1
                self.stats[name]["usage"] = torch.zeros(num_experts)

            for idx in indices.view(-1):
                self.stats[name]["usage"][idx] += 1
                self.stats[name]["total"] += 1

        return hook

    def report(self):
        for name, data in self.stats.items():
            print(f"\n--- 专家分布: {name} ---")
            total = data["total"]
            if total > 0:
                for i, count in enumerate(data["usage"]):
                    print(f"专家 {i}: {count / total:.2%}")