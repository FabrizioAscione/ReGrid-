@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0edit_pareto_plots.py"
if errorlevel 1 pause
