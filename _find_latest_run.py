"""辅助脚本：找到 detection_result/ 下最新的 run 目录并打印路径。"""
import os

RESULT_BASE = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_result"
if os.path.isdir(RESULT_BASE):
    runs = sorted(d for d in os.listdir(RESULT_BASE)
                  if d.startswith("run_") and os.path.isdir(os.path.join(RESULT_BASE, d)))
    if runs:
        print(os.path.join(RESULT_BASE, runs[-1]))
