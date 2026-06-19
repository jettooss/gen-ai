"""
RAG: BM25-поиск по корпусу вакансий + LLM-ответ с цитатами.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent))

if TYPE_CHECKING:
    from schemas import Vacancy

from llm_client import get_model, make_client
from schemas import AnswerWithSources, VacancySummary

EXTRACTED = Path(__file__).parent / "output" / "extracted_vacancies.json"


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9+#.]+", text.lower())


def load_vacancies() -> list[dict]:
    if not EXTRACTED.exists():
        raise FileNotFoundError(f"Нет {EXTRACTED}. Запусти extractor.py")
    return json.loads(EXTRACTED.read_text(encoding="utf-8"))


def build_index(vacancies: list[dict]):
    from rank_bm25 import BM25Okapi

    corpus = []
    for v in vacancies:
        parts = [
            v.get("title", ""),
            v.get("company", ""),
            " ".join(v.get("required_skills", [])),
            v.get("raw_description", "")[:1000],
        ]
        corpus.append(_tokenize(" ".join(parts)))

    return BM25Okapi(corpus)


def search(
    query: str,
    vacancies: list[dict],
    bm25=None,
    top_k: int = 5,
) -> list[dict]:
    if bm25 is None:
        bm25 = build_index(vacancies)
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [vacancies[i] for i, _ in ranked[:top_k]]


def answer_with_citations(
    query: str,
    context_vacancies: list[dict],
    client=None,
) -> AnswerWithSources:
    if client is None:
        client = make_client()

    summaries = []
    for v in context_vacancies:
        skills = ", ".join(v.get("required_skills", [])[:8])
        s_min = v.get("salary_min")
        s_max = v.get("salary_max")
        salary = ""
        if s_min or s_max:
            salary = f"зп: {s_min or '?'}–{s_max or '?'} {v.get('currency','RUR')}"
        summaries.append(
            f"[{v['id']}] {v['title']} ({v['company']}) | {salary} | "
            f"опыт: {v.get('experience_years', 0)} лет | навыки: {skills}"
        )

    context = "\n".join(summaries)
    result = client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты помощник по поиску работы. На основе переданных вакансий "
                    "ответь на вопрос пользователя. Ссылайся на конкретные вакансии "
                    "по их ID в квадратных скобках."
                ),
            },
            {
                "role": "user",
                "content": f"Вакансии:\n{context}\n\nВопрос: {query}",
            },
        ],
        response_model=AnswerWithSources,
        temperature=0.0,
    )
    return result


if __name__ == "__main__":
    vacancies = load_vacancies()
    bm25 = build_index(vacancies)
    query = "Python разработчик с опытом в ML и зарплатой от 200 тысяч"
    top = search(query, vacancies, bm25, top_k=5)
    print(f"Топ-5 по запросу '{query}':")
    for v in top:
        print(f"  [{v['id']}] {v['title']} — {v['company']}")
    answer = answer_with_citations(query, top)
    print(f"\nОтвет: {answer.answer}")
    print(f"Источники: {answer.source_ids}")
