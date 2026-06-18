@echo off
chcp 65001 >nul
cd /d E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier

set PYTHON=E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe

%PYTHON% _find_latest_run.py > %TEMP%\_run_dir.txt
set /p RUN_DIR=<%TEMP%\_run_dir.txt

if "%RUN_DIR%"=="" (
    echo [ERROR] No run folder found. Run full pipeline first.
    pause >nul
    exit /b 1
)

set IMAGES_DIR=%RUN_DIR%\images
set JSON_DIR=%RUN_DIR%\json
set VIZ_DIR=%RUN_DIR%\visualization

if not exist %IMAGES_DIR% (
    echo [ERROR] Images folder not found: %IMAGES_DIR%
    pause >nul
    exit /b 1
)

echo ============================================================
echo  Using run folder: %RUN_DIR%
echo ============================================================
%PYTHON% gbt_step2_detector.py --input %IMAGES_DIR% --output-json %JSON_DIR% --output-viz %VIZ_DIR%

echo.
echo Done. Results: %RUN_DIR%
pause >nul
