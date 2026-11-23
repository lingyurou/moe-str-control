import os
import time
import subprocess
import pandas as pd

CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "eval_history.csv"
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CMD = "{} {} --checkpoint {{}}".format(sys.executable, os.path.join(_HERE, "test.py"))

print(f"🚀 自动化评测中枢已启动，正在监控: {CHECKPOINT_DIR}")

processed_epochs = set()

while True:
    # 扫描目录下所有的 .pth 文件
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth") and "semantic" in f]

    for file in files:
        # 从文件名提取 Epoch 号: g_moe_semantic_epoch_7.pth -> 7
        epoch = file.split("_")[-1].replace(".pth", "")

        if epoch not in processed_epochs:
            print(f"\n✨ 发现新权重: {file}，启动全量评测...")

            # 触发评测
            cp_path = os.path.join(CHECKPOINT_DIR, file)
            result = subprocess.check_output(TEST_CMD.format(cp_path), shell=True, text=True)

            # 简单的正则提取准确率 (根据你之前的 test.py 输出格式调整)
            # 假设我们需要提取 CUTE80 的准确率作为参考点
            import re

            acc_match = re.search(r"CUTE80\s+.*\|\s+认对\(顶刊\):\s+(\d+)", result)
            acc = acc_match.group(1) if acc_match else "N/A"

            # 记录结果
            with open(LOG_FILE, "a") as f:
                f.write(f"{epoch},{acc}\n")

            processed_epochs.add(epoch)
            print(f"✅ Epoch {epoch} 评测完毕，CUTE80 顶刊准确率: {acc}")

    time.sleep(60)  # 每分钟扫描一次