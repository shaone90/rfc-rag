# Веб-сервис: gradio + поиск по numpy + индекс. Моделей внутри нет — они живут
# на домашней машине с видеокартой, контейнер ходит к ним по HTTP через туннель.
FROM python:3.12-slim

# Без этого print() копится в буфере питона и docker logs показывает пустоту.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Зависимости отдельным слоем, до копирования кода: пока requirements.txt
# не менялся, пересборка после правки app.py не запускает pip заново.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Только то, что нужно сервису. index.py и download_corpus.py запускаются дома,
# где лежит корпус и поднят эмбеддер, — в образе им делать нечего.
COPY rag.py app.py ./

# Индекс не вшит в образ, а монтируется томом: он меняется отдельно от кода,
# и пересобирать образ ради новых эмбеддингов незачем.
ENV INDEX_DIR=/data \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Процесс не от root: дыра в gradio упрётся в права обычного пользователя.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 7860

# docker ps будет показывать healthy/unhealthy. curl в slim-образе нет,
# поэтому проверяем питоном, который и так есть.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')" || exit 1

CMD ["python", "app.py"]
