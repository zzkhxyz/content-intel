"""Доводчик пилота: гонит оставшиеся этапы одним отдельным процессом.

1) Whisper по хитам (пока остывает субтитровый лимит YouTube)
2) run_pipeline --from 3: субтитры -> whisper-добор -> темы -> gap -> брифы

Запускается отдельно от интерактивной сессии; лог: logs/finish_pilot.out
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "finish_pilot.out"

FFMPEG_BIN = (Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
              / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
              / "ffmpeg-8.1.2-full_build/bin")


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(args: list[str]) -> int:
    log(f"запускаю: {' '.join(args)}")
    with open(LOG, "a", encoding="utf-8") as f:
        code = subprocess.run([sys.executable] + args, cwd=ROOT,
                              stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"завершено с кодом {code}")
    return code


def main():
    if FFMPEG_BIN.exists():
        os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
    log("=== СТАРТ finish_pilot ===")
    run([str(ROOT / "src" / "04_transcribe.py")])       # хиты, GPU
    run([str(ROOT / "run_pipeline.py"), "--from", "3"])  # субтитры -> ... -> брифы
    log("=== КОНЕЦ finish_pilot ===")


if __name__ == "__main__":
    main()
