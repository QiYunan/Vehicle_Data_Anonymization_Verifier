@echo off
chcp 65001 >nul
cd /d E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier

set PYTHON=E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe

%PYTHON% _make_run_dir.py > %TEMP%\_run_dir.txt
set /p RUN_DIR=<%TEMP%\_run_dir.txt

set IMAGES_DIR=%RUN_DIR%\images
set JSON_DIR=%RUN_DIR%\json
set VIZ_DIR=%RUN_DIR%\visualization

echo ============================================================
echo  Run folder: %RUN_DIR%
echo ============================================================
echo.
echo ============================================================
echo  Step 1/2  Frame extraction
echo ============================================================
%PYTHON% gbt_step1_splitter.py --output %IMAGES_DIR%
if errorlevel 1 (
    echo [ERROR] Step 1 failed.
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo  Step 2/2  Detection
echo ============================================================
%PYTHON% gbt_step2_detector.py --input %IMAGES_DIR% --output-json %JSON_DIR% --output-viz %VIZ_DIR%
if errorlevel 1 (
    echo [ERROR] Step 2 failed.
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline done. Results: %RUN_DIR%
echo ============================================================
pause >nul
