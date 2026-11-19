@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo ReGrid virtual environment not found.
    echo.
    echo Create it with Python 3.10 and install the requirements first:
    echo   py -3.10 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
)
