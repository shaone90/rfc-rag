r"""
Консольная версия: вопрос -> ответ со ссылками.
Вся логика лежит в rag.py, здесь только ввод-вывод.

Запуск:  .venv\Scripts\python.exe ask.py
Выход:   пустая строка или Ctrl+C
"""

import rag


def main():
    vectors, chunks = rag.load_index()

    print(f"Индекс: {len(chunks)} кусков. Порог отсечки: {rag.MIN_SCORE}")
    print("Спрашивай (пустая строка — выход).\n")

    while True:
        try:
            question = input("вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        print("\n  думаю...", end="", flush=True)
        try:
            text, hits, best = rag.answer(question, vectors, chunks)
        except Exception as e:
            print(f"\rОшибка обращения к LM Studio: {e}\n")
            continue
        print("\r" + " " * 12 + "\r", end="")

        print(text)

        if hits:
            print("\nИсточники:")
            for i, (chunk, score) in enumerate(hits, 1):
                print(f"  [{i}] {chunk['source']}  фрагмент #{chunk['n']}  близость {score:.3f}")
        print()


if __name__ == "__main__":
    main()
