@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin;%PATH%"
if not exist logs mkdir logs

echo === СТАРТ finish_pilot: %date% %time% === >> logs\finish_pilot.out

rem 1) Whisper по хитам (субтитровый лимит YouTube тем временем остывает)
python src\04_transcribe.py >> logs\finish_pilot.out 2>&1

rem 2) Дальше вся цепочка: субтитры (остывший лимит) -> whisper-добор -> анализ
python run_pipeline.py --from 3 >> logs\finish_pilot.out 2>&1

echo === КОНЕЦ finish_pilot: %date% %time% === >> logs\finish_pilot.out
