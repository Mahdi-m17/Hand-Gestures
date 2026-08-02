@echo off
title Gesture Playground Share Tunnel
cd /d "%~dp0"
echo Starting local server on port 8765...
start "gesture-http" /MIN cmd /c python -m http.server 8765 --directory "%~dp0"
timeout /t 2 /nobreak >nul
echo Starting public HTTPS tunnel...
echo Share the trycloudflare.com URL that appears below.
echo Keep this window open while others are testing.
echo.
"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:8765 --no-autoupdate
