# Пайплайн Content Intelligence.
# ВАЖНО: в контейнере Whisper (этап 04) работает на CPU — для GPU нужен
# WSL2 + NVIDIA container toolkit + docker run --gpus all.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt \
    && pip install --no-cache-dir -U yt-dlp

COPY src/ src/
COPY run_pipeline.py .

# config/, data/, output/, logs/ монтируются томами (см. docker-compose.yml),
# чтобы правка каналов/настроек не требовала пересборки образа,
# а данные и результаты жили на хосте.
CMD ["python", "run_pipeline.py"]
