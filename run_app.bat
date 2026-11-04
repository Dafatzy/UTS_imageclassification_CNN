@echo off
cd /d "%~dp0"
echo Starting Landscape AI Classifier...
py -m streamlit run app.py
pause
