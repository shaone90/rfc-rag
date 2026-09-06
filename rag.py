r"""
Ядро проекта: загрузка индекса, поиск по эмбеддингам, генерация ответа.
Импортируется из ask.py (консоль) и app.py (веб).
"""

import json
import os
import sys

import numpy as np
import requests

# --- настройки ---------------------------------------------------------------

LMS_URL = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-bge-m3"
CHAT_MODEL = "gemma-4-e4b-it-ultra-uncensored-heretic"     # <-- подставь свой id из /v1/models

TOP_K = 8           # сколько кусков отдаём модели
MIN_SCORE = 0.55    # ниже этой близости считаем, что в корпусе ответа нет
MAX_CHARS = 1500    # обрезка куска перед отправкой в модель

VECTORS_PATH = "index_vectors.npy"
CHUNKS_PATH = "index_chunks.json"

PROMPT_TEMPLATE = """Ты отвечаешь на вопросы строго по приведённым фрагментам документов.

Правила:
- Используй только информацию из фрагментов ниже. Ничего не добавляй от себя.
- После каждого утверждения ставь номер фрагмента в квадратных скобках, например [2].
- Если фрагментов не хватает для ответа, так и скажи. Не выдумывай.
- Если фрагменты о смежной теме, но на сам вопрос не отвечают, так и скажи.
  Не пересказывай соседнее вместо ответа.
- Отвечай на языке вопроса, кратко и по делу.

Фрагменты:
{sources}

Вопрос: {question}
"""

SESSION = requests.Session()   # переиспользуем TCP-соединение вместо нового на каждый запрос

# -----------------------------------------------------------------------------


def load_index():
    """Читает матрицу векторов и тексты кусков. Возвращает (vectors, chunks)."""
    if not os.path.exists(VECTORS_PATH):
        print("Индекса нет. Сначала запусти index.py")
        sys.exit(1)

    vectors = np.load(VECTORS_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return vectors, chunks


def embed_one(text):
    resp = SESSION.post(
        f"{LMS_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def generate(prompt):
    resp = SESSION.post(
        f"{LMS_URL}/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def search(question, vectors, chunks, top_k=TOP_K):
    """Возвращает список (кусок, близость), отсортированный по убыванию."""
    qv = np.array(embed_one(question), dtype="float32")
    qv /= np.linalg.norm(qv)

    scores = vectors @ qv                  # косинусная близость ко всем кускам сразу
    top = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_prompt(question, hits):
    blocks = []
    for i, (chunk, _score) in enumerate(hits, 1):
        body = " ".join(chunk["text"].split())[:MAX_CHARS]
        blocks.append(f"[{i}] источник: {chunk['source']}, фрагмент #{chunk['n']}\n{body}")
    return PROMPT_TEMPLATE.format(sources="\n\n".join(blocks), question=question)


def answer(question, vectors, chunks):
    """Полный цикл. Возвращает (текст ответа, список попаданий, лучшая близость).

    Если лучшая близость ниже порога, модель не вызывается вообще —
    это экономит время и не даёт пересказывать заведомо нерелевантное.
    """
    hits = search(question, vectors, chunks)
    best = hits[0][1]

    if best < MIN_SCORE:
        text = (f"В документах нет ответа на этот вопрос "
                f"(лучшая близость {best:.3f} < {MIN_SCORE}).")
        return text, [], best

    return generate(build_prompt(question, hits)).strip(), hits, best
