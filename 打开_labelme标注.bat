@echo off
chcp 65001 >nul
rem 打开 labelme 标注工具。可把要标注的图片文件夹拖到本 bat 上，或启动后在界面里「打开目录」。
rem GT 真值 JSON 默认保存在图片旁边（与图片同名 .json）。
echo Starting Labelme, please wait 5-10 seconds for the window to appear...
"E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\labelme.exe" %*
