@echo off
chcp 65001 >nul
cd /d "E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier"
echo [Script 5] Self-check evaluation ...
"E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe" gbt_step5_evaluator.py
echo.
echo Done. Press any key to close.
pause >nul
