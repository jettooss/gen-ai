# Финальный проект: Умный поиск IT-вакансий

**Трек B — прикладной проект**  
**Дедлайн:** 22.06.2026

## Задача

По профилю кандидата (навыки, опыт, зарплатные ожидания) найти подходящие
IT-вакансии, оценить совпадение и указать пробелы в навыках.

**Данные:** 100 IT-вакансий из датасета
[IT vacancies from hh.ru](https://huggingface.co/datasets/trewwxsav/IT_vacancies_from_hh.ru)
на HuggingFace (68 584 реальных вакансий hh.ru).

## Техники курса

| # | Техника | Файл |
|---|---------|------|
| 1 | Structured output + `@field_validator` | `schemas.py`, `extractor.py` |
| 2 | RAG (BM25 + LLM answer gen) | `rag.py` |
| 3 | LLM-as-judge (hallucination check + eval) | `hallucination.py`, `eval.py` |
| 4 | Агент с инструментами (tools: search, score, gap) | `agent.py` |
| 5 | Мультиагент (Planner → Workers → Critic) | `agent.py` |

## Запуск

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Настроить LLM
cp .env.example .env
# вписать OPENAI_API_KEY и OPENAI_BASE_URL

# 3. Загрузить данные (CSV уже в input/, либо скачать вручную)
python scraper.py           # -> input/vacancies.json (100 вакансий)

# 4. Извлечь структурированные данные
python extractor.py         # -> output/extracted_vacancies.json

# 5. Проверить галлюцинации
python hallucination.py     # -> output/hallucination_report.json

# 6. Оценить систему (15 кейсов)
python eval.py              # -> output/eval_results.json
```

## Структура

```
финальный_проект/
├── llm_client.py        — LLM клиент (JSON-инструктор)
├── schemas.py           — Pydantic-модели с field_validator
├── scraper.py           — загрузка вакансий с hh.ru API
├── extractor.py         — structured extraction через LLM
├── rag.py               — BM25 + LLM с цитатами
├── hallucination.py     — LLM-as-judge проверка навыков
├── agent.py             — мультиагент Planner/Worker/Critic
├── eval.py              — 15 тест-кейсов, pass-rate
├── input/vacancies.json — исходные данные
├── output/              — результаты прогонов
└── отчёт.md             — отчёт с реальными числами
```

## Результаты

См. [отчёт.md](отчёт.md).
