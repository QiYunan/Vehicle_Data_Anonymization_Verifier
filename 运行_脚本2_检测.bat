@echo off
chcp 65001 >nul
cd /d "E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier"
echo [Script 2] Detecting faces / plates ...
"E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe" gbt_step2_detector.py
echo.
echo Done. Press any key to close.
pause >nul
