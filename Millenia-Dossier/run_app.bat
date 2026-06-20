@echo off
setlocal
cd /d %~dp0

echo Starting Millenia Dossier...
echo.
echo If your Python environment is not active, activate it first.
echo.

streamlit run app\streamlit_app.py
pause
