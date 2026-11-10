@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_qwen_smoke.ps1"
echo.
echo The smoke test has stopped. Keep this window open and tell Codex it is finished.
pause
endlocal
