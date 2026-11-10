@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_qwen_full_1950.ps1"
echo.
echo The full 1950-sample run has stopped. Keep this window open and tell Codex it is finished.
pause
endlocal
