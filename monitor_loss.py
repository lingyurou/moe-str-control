import re
import time

# 训练日志文件路径 (请根据你的实际输出路径修改)
# 如果你是直接在终端看，这里可以修改为实时读取管道，但最简单的办法是让 train.py 把日志写入文件
LOG_FILE = "train_log.txt"
OUTPUT_CSV = "loss_history.csv"

print(f"🔍 监控已启动，正在监听 {LOG_FILE}...")

with open(OUTPUT_CSV, "w") as f_out:
    f_out.write("epoch,batch,loss,ce_loss,aux_loss\n")

# 使用正则提取 Loss 信息
pattern = re.compile(r"Epoch: (\d+) \[.*? (\d+)/.*?\] \| Loss: ([\d\.]+) \(CE: ([\d\.]+), Aux: ([\d\.]+)\)")

try:
    with open(LOG_FILE, "r") as f:
        f.seek(0, 2)  # 跳到文件末尾
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue

            match = pattern.search(line)
            if match:
                epoch, batch, loss, ce, aux = match.groups()
                print(f"✅ 捕获: Epoch {epoch} Batch {batch} | Loss: {loss}")
                with open(OUTPUT_CSV, "a") as f_out:
                    f_out.write(f"{epoch},{batch},{loss},{ce},{aux}\n")
except KeyboardInterrupt:
    print("🛑 监控已停止。")