r"""
Читает corpus/*.txt, режет на куски, считает эмбеддинги через LM Studio
и складывает результат в два файла: index_vectors.npy и index_chunks.json.

Перед запуском: в LM Studio должен быть поднят локальный сервер
и загружена embedding-модель.

Запуск:  .venv\Scripts\python.exe index.py
"""

import glob
import json
import os
import re
import sys
import time

import numpy as np
import requests

# --- настройки ---------------------------------------------------------------

# ВАЖНО: имя модели должно совпадать с тем, что отдаёт LM Studio.
# Открой в браузере http://localhost:1234/v1/models и скопируй оттуда "id".
LMS_URL = os.environ.get("LMS_URL", "http://localhost:1234/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-bge-m3")

CHUNK_SIZE = 1500   # примерный размер куска в символах
BATCH = 32          # сколько кусков отправляем в модель за раз

# -----------------------------------------------------------------------------


def clean(text):
    """Выкидывает колонтитулы RFC — иначе они лезут в каждый кусок и портят поиск."""
    text = text.replace("\x0c", "\n")   # символ разрыва страницы
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if re.search(r"\[Page \d+\]$", s):          # "... [Page 12]"
            continue
        if re.match(r"^RFC \d+.*\d{4}$", s):        # "RFC 791  Internet Protocol  September 1981"
            continue
        out.append(line.rstrip())
    return "\n".join(out)


def split_chunks(text):
    """Режет текст по абзацам, набирая куски примерно по CHUNK_SIZE символов.
    Последний абзац переносится в начало следующего куска — это перекрытие,
    чтобы мысль не обрывалась ровно на границе."""
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        while len(p) > CHUNK_SIZE:      # таблицы в RFC идут сплошняком без пустых строк
            paras.append(p[:CHUNK_SIZE])
            p = p[CHUNK_SIZE:]
        if p:
            paras.append(p)
    chunks = []
    cur = ""

    for p in paras:
        if cur and len(cur) + len(p) + 2 > CHUNK_SIZE:
            chunks.append(cur)
            tail = cur.split("\n\n")[-1]
            cur = (tail + "\n\n" + p) if len(tail) < 400 else p
        else:
            cur = (cur + "\n\n" + p) if cur else p

    if cur:
        chunks.append(cur)
    return chunks


def embed(texts):
    """Отправляет список текстов в LM Studio, получает список векторов."""
    resp = requests.post(
        f"{LMS_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    if resp.status_code != 200:
        print(f"\nLM Studio ответил {resp.status_code}: {resp.text[:400]}")
        sys.exit(1)

    data = resp.json()["data"]
    data.sort(key=lambda d: d["index"])   # порядок гарантировать не мешает
    return [d["embedding"] for d in data]


def main():
    files = sorted(glob.glob(os.path.join("corpus", "*.txt")))
    if not files:
        print("В папке corpus/ пусто. Сначала запусти download_corpus.py")
        sys.exit(1)

    chunks = []
    for path in files:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        parts = split_chunks(clean(raw))
        for i, part in enumerate(parts):
            chunks.append({"source": name, "n": i, "text": part})
        print(f"  {name:<14} {len(parts)} кусков")

    print(f"\nВсего кусков: {len(chunks)}. Считаю эмбеддинги, это займёт несколько минут.\n")

    vectors = []
    started = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = [c["text"] for c in chunks[i:i + BATCH]]
        vectors.extend(embed(batch))

        done = min(i + BATCH, len(chunks))
        speed = done / max(time.time() - started, 0.1)
        left = (len(chunks) - done) / max(speed, 0.1)
        print(f"\r  {done}/{len(chunks)}   осталось ~{int(left)} с", end="")

    print()

    mat = np.array(vectors, dtype="float32")
    # нормируем векторы: после этого косинусная близость = обычное скалярное произведение
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    np.save("index_vectors.npy", mat)
    with open("index_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"\nГотово. Матрица {mat.shape[0]} x {mat.shape[1]}, "
          f"{mat.nbytes // 1024} КБ. Теперь запускай search.py")


if __name__ == "__main__":
    main()
