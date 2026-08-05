@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set UNIFIED_MEMORY_STORAGE_DIR=%~dp0data
set UNIFIED_MEMORY_CONFIG=%~dp0config.json
"%~dp0..\..\ji\work\python311\python.exe" "%~dp0run_http.py"
pause
