r"""
Веб-морда на Gradio.

Запуск:  .venv\Scripts\python.exe app.py
Потом открыть в браузере http://127.0.0.1:7860
"""

import gradio as gr

import rag

VECTORS, CHUNKS = rag.load_index()

EXAMPLES = [
    "зачем в tcp нужен трёхэтапный хендшейк",
    "какие диапазоны адресов зарезервированы под частные сети",
    "что такое neighbor discovery в ipv6",
    "какие типы сообщений бывают в bgp",
    "как мне включить вайфай",
]


def on_ask(question):
    question = (question or "").strip()
    if not question:
        return "Введи вопрос.", ""

    try:
        text, hits, best = rag.answer(question, VECTORS, CHUNKS)
    except Exception as e:
        return f"Ошибка обращения к LM Studio: {e}", ""

    if not hits:
        return text, ""

    rows = ["| № | Документ | Фрагмент | Близость |", "|---|---|---|---|"]
    for i, (chunk, score) in enumerate(hits, 1):
        rows.append(f"| {i} | {chunk['source']} | #{chunk['n']} | {score:.3f} |")

    return text, "\n".join(rows)


with gr.Blocks(title="RAG по RFC") as demo:
    gr.Markdown(
        f"# Поиск по RFC с локальной LLM\n"
        f"Вопрос на любом языке → семантический поиск по "
        f"{len(CHUNKS)} фрагментам из 33 RFC → ответ со ссылками на источники.\n\n"
        f"Всё считается локально: эмбеддер `{rag.EMBED_MODEL}` и "
        f"модель `{rag.CHAT_MODEL}` в LM Studio."
    )

    question = gr.Textbox(
        label="Вопрос",
        placeholder="например: чем отличается tls 1.3 от предыдущих версий",
        lines=2,
    )
    ask_btn = gr.Button("Спросить", variant="primary")

    answer_box = gr.Markdown(label="Ответ")
    gr.Markdown("### Источники")
    sources_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=question, label="Примеры")

    ask_btn.click(on_ask, inputs=question, outputs=[answer_box, sources_box])
    question.submit(on_ask, inputs=question, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
