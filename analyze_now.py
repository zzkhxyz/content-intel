"""Быстрый анализ по уже собранным данным: этапы 5-7 (темы -> gap -> брифы).

Запускается отдельным процессом; лог: logs/analyze_now.out
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "analyze_now.out"


def main():
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"=== СТАРТ analyze_now {time.strftime('%H:%M:%S')} ===\n")
        f.flush()
        code = subprocess.run(
            [sys.executable, str(ROOT / "run_pipeline.py"), "--from", "5"],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
        ).returncode
        f.write(f"=== КОНЕЦ analyze_now, код {code}, {time.strftime('%H:%M:%S')} ===\n")


if __name__ == "__main__":
    main()
