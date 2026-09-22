@echo off
REM CropGuardAI Training Dashboard Launcher (Windows)
REM Usage: run_dashboard.bat

cd /d "%~dp0.."

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing streamlit...
    pip install streamlit --quiet
)

echo.
echo ==========================================
echo  CropGuardAI Training Dashboard
echo ==========================================
echo.
echo Starting Streamlit server...
echo Open http://localhost:8501 in your browser
echo.
echo Press Ctrl+C to stop
echo.

streamlit run scripts/training_dashboard.py --server.port 8501 --server.headless true