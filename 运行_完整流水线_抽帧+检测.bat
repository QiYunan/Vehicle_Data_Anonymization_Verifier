@echo off
chcp 65001 >nul
cd /d "E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier"

echo ============================================================
echo  Step 1/2  Frame extraction (1 frame per 2 sec)
echo ============================================================
"E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe" gbt_step1_splitter.py
if errorlevel 1 (
    echo [ERROR] Step 1 failed. Check video path or ffmpeg.
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo  Step 2/2  Face and plate detection
echo ============================================================
"E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe" gbt_step2_detector.py
if errorlevel 1 (
    echo [ERROR] Step 2 failed.
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline done.
echo  Results: E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_json\
echo ============================================================
pause >nul
