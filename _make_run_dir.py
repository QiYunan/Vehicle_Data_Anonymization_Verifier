"""辅助脚本：生成下一个 run 目录路径并打印到 stdout，供 bat 文件捕获。"""
import os
from datetime import datetime

RESULT_BASE = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_result"
os.makedirs(RESULT_BASE, exist_ok=True)

max_n = 0
for name in os.listdir(RESULT_BASE):
    if name.startswith("run_") and os.path.isdir(os.path.join(RESULT_BASE, name)):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            max_n = max(max_n, int(parts[1]))

run_name = f"run_{max_n + 1}_{datetime.now().strftime('%Y%m%d_%H%M')}"
print(os.path.join(RESULT_BASE, run_name))
