@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0save_dashscope_credentials.ps1"
echo.
pause
endlocal
