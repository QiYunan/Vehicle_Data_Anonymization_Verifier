@echo off
chcp 65001 >nul
cd /d E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier

REM ============================================================
REM  统一检测入口（照片 + 视频 混选）
REM  弹窗勾选要检测的照片/视频 -> 视频抽帧 + 照片直采 -> 脚本二检测
REM  结果: test\detection_result\run_N_timestamp\{json,visualization}
REM ============================================================

set PYTHON=E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe

%PYTHON% gbt_run_detection.py

echo.
pause >nul
