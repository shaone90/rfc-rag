r"""
Консольный поиск по индексу. Вводишь вопрос — получаешь 5 самых близких кусков
с указанием, из какого RFC они взяты.

Пока без генерации ответа: сначала надо убедиться, что поиск находит нужное.
Если тут мусор — LLM его только красиво перескажет, и толку не будет.

Запуск:  .venv\Scripts\python.exe search.py
Выход:   пустая строка или Ctrl+C
"""

import json
import os
import sys

import numpy as np
import requests

LMS_URL = os.environ.get("LMS_URL", "http://localhost:1234/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-bge-m3")   # то же, что в index.py
TOP_K = 5
PREVIEW = 600   # сколько символов куска показывать


def embed_one(text):
    resp = requests.post(
        f"{LMS_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"LM Studio ответил {resp.status_code}: {resp.text[:400]}")
        sys.exit(1)
    return resp.json()["data"][0]["embedding"]


def main():
    if not os.path.exists("index_vectors.npy"):
        print("Индекса нет. Сначала запусти index.py")
        sys.exit(1)

    vectors = np.load("index_vectors.npy")
    with open("index_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Загружен индекс: {len(chunks)} кусков. Спрашивай (пустая строка — выход).\n")

    while True:
        try:
            question = input("вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        qv = np.array(embed_one(question), dtype="float32")
        qv /= np.linalg.norm(qv)

        scores = vectors @ qv                      # косинусная близость ко всем кускам сразу
        top = np.argsort(-scores)[:TOP_K]

        print()
        for rank, idx in enumerate(top, 1):
            c = chunks[idx]
            body = " ".join(c["text"].split())     # схлопываем переносы, чтобы читалось в консоли
            if len(body) > PREVIEW:
                body = body[:PREVIEW] + "..."
            print(f"[{rank}] {c['source']}  кусок #{c['n']}  близость {scores[idx]:.3f}")
            print(f"    {body}\n")


if __name__ == "__main__":
    main()
