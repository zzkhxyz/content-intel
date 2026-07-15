@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Content Intelligence - запуск пайплайна
echo ============================================

where python >nul 2>nul || (echo Python не найден в PATH. Установи Python 3.11+ и перезапусти. & pause & exit /b 1)

if not exist .venv (
    echo Создаю виртуальное окружение...
    python -m venv .venv || (echo Не удалось создать venv & pause & exit /b 1)
)

call .venv\Scripts\activate.bat
python -m pip install -q -U -r requirements.txt || (echo Не удалось поставить зависимости & pause & exit /b 1)
python -m pip install -q -U yt-dlp

python run_pipeline.py %*

echo.
pause
