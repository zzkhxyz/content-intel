"""Оркестратор пайплайна: запускает этапы 01-07 по порядку, стоп на ошибке.

Запуск:
    python run_pipeline.py              # все этапы
    python run_pipeline.py --from 3     # с этапа 3 (после падения/лимита)
    python run_pipeline.py --to 3       # только этапы 1-3 (без LLM-ключа)
    python run_pipeline.py --only 6     # один этап

Каждый этап кэширует результаты, поэтому повторный полный прогон быстрый:
уже сделанное не переделывается и не тратит лимиты API.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("01_harvest.py",    "Сбор метаданных каналов"),
    ("02_score.py",      "Расчёт метрик (outlier_score, хиты)"),
    ("03_subtitles.py",  "Авто-субтитры"),
    ("04_transcribe.py", "Whisper по хитам"),
    ("05_topics.py",     "Темы + кластеризация (LLM)"),
    ("06_gap.py",        "GAP-АНАЛИЗ (LLM + поиск YouTube)"),
    ("07_briefs.py",     "Контент-брифы (LLM)"),
]


def preflight(first: int, last: int) -> None:
    """Быстрые проверки, чтобы упасть в первую секунду, а не через час."""
    problems = []
    channels = (ROOT / "config" / "channels.yaml")
    if first <= 1 and not channels.exists():
        problems.append("нет config/channels.yaml")
    if last >= 5 and not os.getenv("GROQ_API_KEY"):
        # utils подхватывает .env сам — здесь дублируем его логику для проверки
        for env_path in (ROOT / ".env", ROOT.parent / ".env"):
            if env_path.exists() and "GROQ_API_KEY" in env_path.read_text(
                    encoding="utf-8", errors="replace"):
                break
        else:
            problems.append(
                "GROQ_API_KEY не найден (нужен с этапа 5): впиши ключ в .env "
                "или запусти пока только сбор: python run_pipeline.py --to 4")
    if problems:
        sys.exit("Проверка перед запуском не пройдена:\n  - " + "\n  - ".join(problems))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_", type=int, default=1, metavar="N")
    parser.add_argument("--to", type=int, default=len(STAGES), metavar="N")
    parser.add_argument("--only", type=int, default=None, metavar="N")
    args = parser.parse_args()

    first = args.only or args.from_
    last = args.only or args.to
    preflight(first, last)

    t0 = time.time()
    for i, (script, title) in enumerate(STAGES, start=1):
        if not (first <= i <= last):
            continue
        print(f"\n{'=' * 70}\n  ЭТАП {i}/7: {title}\n{'=' * 70}", flush=True)
        t_stage = time.time()
        code = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                              cwd=ROOT).returncode
        mins = (time.time() - t_stage) / 60
        if code != 0:
            print(f"\nЭТАП {i} УПАЛ (код {code}, {mins:.1f} мин). "
                  f"Смотри logs/{script.replace('.py', '')}.log")
            print(f"После починки продолжить с этого места:\n"
                  f"    python run_pipeline.py --from {i}")
            sys.exit(code)
        print(f"  этап {i} готов за {mins:.1f} мин")

    print(f"\n{'=' * 70}\n  ПАЙПЛАЙН ЗАВЕРШЁН за {(time.time() - t0) / 60:.0f} мин")
    print("  Деливерабл: output/gap_analysis.png + output/briefs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
