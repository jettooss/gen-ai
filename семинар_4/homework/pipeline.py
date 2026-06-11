import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STRATEGY = "smart"
QUESTION = "Как RAG снижает риск галлюцинаций?"


def tokenize(text):
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def fixed_split(text):
    chunks = []
    for i in range(0, len(text), 2000):
        chunk = text[i : i + 2000].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def smart_split(text):
    chunks = []
    paragraphs = re.split(r"\n\s*\n", text)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > 400:
            chunks.append(paragraph[:400].strip())
            paragraph = paragraph[320:].strip()  # overlap 80 символов

        if paragraph:
            chunks.append(paragraph)

    return chunks


def make_chunks(strategy):
    chunks = []
    splitter = fixed_split if strategy == "fixed" else smart_split

    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for number, chunk_text in enumerate(splitter(text)):
            chunks.append(
                {
                    "id": f"{path.stem}__{number}",
                    "source": path.stem,
                    "text": chunk_text,
                    "tokens": tokenize(chunk_text),
                }
            )

    return chunks


def score(question_tokens, chunk_tokens):
    counts = Counter(chunk_tokens)
    result = 0

    for token in question_tokens:
        result += counts[token]

    # небольшой бонус, если несколько слов запроса идут подряд
    question_text = " ".join(question_tokens)
    chunk_text = " ".join(chunk_tokens)
    if question_text and question_text in chunk_text:
        result += 3

    return result


def retrieve(question, strategy="fixed", k=5):
    question_tokens = tokenize(question)
    results = []

    for chunk in make_chunks(strategy):
        results.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "score": score(question_tokens, chunk["tokens"]),
                "text": chunk["text"],
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def ingest(strategy):
    chunks = make_chunks(strategy)
    sources = {chunk["source"] for chunk in chunks}
    print(f"{strategy}: {len(chunks)} чанков из {len(sources)} документов")


def ask(question, strategy):
    print(f"Вопрос: {question}")
    print(f"Стратегия: {strategy}\n")

    for hit in retrieve(question, strategy):
        preview = " ".join(hit["text"].split())[:350]
        print(f"{hit['id']} | score={hit['score']}")
        print(preview)
        print()


if __name__ == "__main__":
    ingest("fixed")
    ingest("smart")
    print()
    ask(QUESTION, STRATEGY)
