@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Content Intelligence - запуск в Docker
echo ============================================

docker info >nul 2>nul || (echo Docker не запущен. Запусти Docker Desktop и повтори. & pause & exit /b 1)

docker compose build || (echo Сборка образа упала & pause & exit /b 1)
docker compose run --rm pipeline python run_pipeline.py %*

echo.
pause
