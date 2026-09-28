@echo off
cd /d "%~dp0"
echo ========================================================
echo   Starting YOLO11-Pose Posture Analysis System...
echo ========================================================
python -m streamlit run app.py
pause
